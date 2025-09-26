import os
import re
import json
import requests
from dotenv import load_dotenv
from typing import List, TypedDict

from fastapi import FastAPI, Request, HTTPException, Response
# from pydantic import BaseModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END

from requests.exceptions import HTTPError

# 1. chamando a API KEY
load_dotenv()
if os.getenv("GOOGLE_API_KEY") is None:
    print("Erro: A chave de API do Google não foi encontrada.")
    exit()
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.2) # temperatura baixa para ser menos criativo e mais consistente

# --- NOSSO TESTE DE DEPURACAO ---
print("--- MODELO CONFIGURADO ---")
print(f"--> O modelo carregado no código é: {llm.model}")
print("--------------------------")

WHATSAPP_API_TOKEN = os.getenv("WHATSAPP_API_TOKEN")
WHATSAPP_PHONE_NUMBER_ID = os.getenv("WHATSAPP_PHONE_NUMBER_ID")
VERIFY_TOKEN = os.getenv("VERIFY_TOKEN")

# prompt do sistema e palavras-chave de emergência
SYSTEM_PROMPT = """
Você é um assistente de triagem virtual da ClinicAI. 
Sua persona é acolhedora, empática, calma e profissional. 
Use linguagem clara, simples e direta, sem jargões médicos. 
Seja humanizado e acolhedor, mas não excessivamente informal.

SEMPRE SIGA ESTAS REGRAS:

1. **APRESENTAÇÃO INICIAL:** 
Na primeira mensagem, apresente-se como um assistente virtual da ClinicAI, explique que seu objetivo é coletar informações para agilizar a consulta e DEIXE CLARO que você não substitui a avaliação de um profissional de saúde.

2. **MISSÃO - COLETAR DADOS:** 
Conduza a conversa para coletar os seguintes dados de forma estruturada: 
- Queixa Principal 
- Sintomas Detalhados 
- Duração e Frequência 
- Intensidade da dor (0 a 10) 
- Histórico Relevante 
- Medidas já tomadas

3. **ARMAZENAMENTO:** 
Ao final da triagem, organize as informações em um resumo estruturado 
e armazene-o de forma consultável, vinculado ao número de telefone do usuário (hasheado ou não).

4. **PROTOCOLO DE EMERGÊNCIA:** 
Se o usuário mencionar qualquer sintoma ou palavra-chave de emergência (ou sinônimos diretos), 
INTERROMPA a triagem imediatamente e responda APENAS com a mensagem de emergência.

   - PALAVRAS-CHAVE DE EMERGÊNCIA: 
     dor no peito, aperto no peito, falta de ar, dificuldade para respirar, desmaio, perda de consciência, 
     convulsão, confusão mental, fraqueza súbita, não mexer um lado do corpo, fala enrolada, visão turva ou perda de visão, 
     dor de cabeça súbita e intensa, sangramento intenso, hemorragia, sangramento na gravidez, dor abdominal súbita intensa, 
     reação alérgica grave, garganta fechando.

   - MENSAGEM DE EMERGÊNCIA: 
   "Entendi. Seus sintomas podem indicar uma situação de emergência. 
   Por favor, procure o pronto-socorro mais próximo ou ligue para o 192 (SAMU) imediatamente."

5. **LIMITAÇÕES CRÍTICAS (O QUE NÃO FAZER):** 
NUNCA ofereça diagnósticos. 
NUNCA sugira tratamentos ou medicamentos. 
Seu papel é apenas triagem inicial e orientação.
"""

# lista determinística para detectar casos de emergência sem precisar chamar o LLM
EMERGENCY_KEYWORDS = [
    "dor no peito", "aperto no peito", "falta de ar", "dificuldade para respirar", "desmaio", "perda de consciência", 
    "convulsão", "confusão mental", "fraqueza súbita", "não mexer um lado do corpo", "fala enrolada", "visão turva ou perda de visão", 
    "dor de cabeça súbita e intensa", "sangramento intenso", "hemorragia", "sangramento na gravidez", "dor abdominal súbita intensa", 
    "reação alérgica grave", "garganta fechando"
]

class AgentState(TypedDict):
    messages: List[BaseMessage]


def call_standard_model(state: AgentState): # resposta padrão
    system_message = SystemMessage(content=SYSTEM_PROMPT)
    messages_with_system_prompt = [system_message] + state['messages']
    response = llm.invoke(messages_with_system_prompt)
    return {"messages": [response]}

def handle_emergency(state: AgentState): # resposta automática de emergência
    emergency_response_text = "Entendi. Seus sintomas podem indicar uma situação de emergência. Por favor, procure o pronto-socorro mais próximo ou ligue para o 192 (SAMU) imediatamente."
    return {"messages": [AIMessage(content=emergency_response_text)]}

def route_message(state: AgentState): # verifica a mensagem e decide o nó
    last_message = state['messages'][-1].content.lower()
    # expressão regular para encontrar qualquer uma das palavras-chave
    if any(re.search(r'\b' + keyword + r'\b', last_message) for keyword in EMERGENCY_KEYWORDS):
        return "emergency"
    return "standard"

workflow = StateGraph(AgentState)

# nós de decisão
workflow.add_node("standard_agent", call_standard_model)
workflow.add_node("emergency_handler", handle_emergency)

# nó de entrada ligado com os dois nós possíveis
workflow.set_conditional_entry_point(
    route_message,
    {
        "standard": "standard_agent",
        "emergency": "emergency_handler",
    }
)
workflow.add_edge("standard_agent", END)
workflow.add_edge("emergency_handler", END)

agent_app = workflow.compile()

# Inicializa o FastAPI
app = FastAPI()

# armazenamento temporário de histórico em memória.
# futuramente será integrado com o MongoDB.
conversation_histories = {}

def send_whatsapp_message(to_number: str, message: str):
    """Função para enviar uma mensagem de volta para o usuário via WhatsApp Cloud API."""
    url = f"https://graph.facebook.com/v19.0/{WHATSAPP_PHONE_NUMBER_ID}/messages"
    headers = {
        "Authorization": f"Bearer {WHATSAPP_API_TOKEN}",
        "Content-Type": "application/json",
    }
    data = {
        "messaging_product": "whatsapp",
        "to": to_number,
        "text": {"body": message},
    }
    
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status() # Lança um erro se a requisição falhar
        print("Mensagem enviada com sucesso!") # Adicionamos um log de sucesso
    except HTTPError as http_err:
        # ESTA É A PARTE NOVA E IMPORTANTE
        print(f"Erro HTTP ao enviar mensagem: {http_err}")
        # Tentamos imprimir a resposta detalhada do servidor do Facebook
        try:
            error_details = http_err.response.json()
            print(f"Detalhes do erro do Facebook: {json.dumps(error_details, indent=2)}")
        except json.JSONDecodeError:
            print(f"Não foi possível decodificar a resposta do erro: {http_err.response.text}")
    except Exception as err:
        print(f"Outro erro ocorreu: {err}")

@app.get("/webhook")
async def verify_webhook(request: Request):
    """Endpoint para a verificação inicial do Webhook pelo Facebook."""
    mode = request.query_params.get("hub.mode")
    token = request.query_params.get("hub.verify_token")
    challenge = request.query_params.get("hub.challenge")

    if mode == "subscribe" and token == VERIFY_TOKEN:
        print(f"Webhook verificado com sucesso!")
        return Response(content=challenge, media_type="text/plain")
    
    raise HTTPException(status_code=403, detail="Falha na verificação do Webhook")

@app.post("/webhook")
async def handle_webhook(request: Request):
    """Endpoint para receber as mensagens do WhatsApp."""
    data = await request.json()
    print("Dados recebidos do WhatsApp:", json.dumps(data, indent=2)) # Log para depuração

    try:
        # Extrai as informações relevantes da mensagem
        # A estrutura pode variar, esta é a mais comum para mensagens de texto
        if "entry" in data and data["entry"][0]["changes"][0]["value"]["messages"]:
            message_data = data["entry"][0]["changes"][0]["value"]["messages"][0]
            sender_id = message_data["from"]
            user_message = message_data["text"]["body"]

            print(f"Mensagem de {sender_id}: {user_message}")

            # Lógica do agente (muito parecida com a da Fase 2)
            if sender_id not in conversation_histories:
                initial_message = AIMessage(content="Olá! Sou a assistente virtual da ClinicAI. Meu objetivo é coletar algumas informações para agilizar sua consulta. É importante lembrar que não substituo a avaliação de um profissional de saúde. Qual o motivo do seu contato hoje?")
                conversation_histories[sender_id] = [initial_message]
                send_whatsapp_message(sender_id, initial_message.content) # Envia a saudação inicial
                return {"status": "ok"}
            
            current_history = conversation_histories[sender_id]
            current_history.append(HumanMessage(content=user_message))
            
            result = agent_app.invoke({"messages": current_history})
            agent_response = result['messages'][-1]
            
            current_history.append(agent_response)
            conversation_histories[sender_id] = current_history
            
            # Envia a resposta do agente de volta para o usuário
            send_whatsapp_message(sender_id, agent_response.content)

    except Exception as e:
        print(f"Erro ao processar a mensagem: {e}")
        # É uma boa prática não retornar um erro para o webhook para evitar que ele seja desativado
        pass

    return {"status": "ok"}