import os
import re
from dotenv import load_dotenv
from typing import List, TypedDict
import datetime

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import pymongo
from pymongo.server_api import ServerApi

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage, messages_from_dict, message_to_dict
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END

# 1. chamando a API KEY
load_dotenv()
if os.getenv("GOOGLE_API_KEY") is None:
    print("Erro: A chave de API do Google não foi encontrada.")
    exit()
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite", temperature=0.2) # temperatura baixa para ser menos criativo e mais consistente
MONGO_URI = os.getenv("MONGO_URI")

try:
    client = pymongo.MongoClient(MONGO_URI, server_api=ServerApi('1'))
    db = client["ClinicAI"] # Nome do nosso banco de dados
    conversations_collection = db["conversations"] # Onde vamos guardar as mensagens
    client.admin.command('ping')
    print("Conexão com o MongoDB Atlas bem-sucedida!")
except Exception as e:
    print(f"Erro ao conectar com o MongoDB: {e}")
    exit()

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

app = FastAPI(title="ClinicAI Agent API")

# permite que a página web (rodando em um local diferente)
# possa fazer requisições para esta API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permite todas as origens (para desenvolvimento)
    allow_credentials=True,
    allow_methods=["*"],  # Permite todos os métodos (GET, POST, etc.)
    allow_headers=["*"],  # Permite todos os cabeçalhos
)

class ChatRequest(BaseModel):
    sender_id: str
    message: str

@app.post("/chat")
def process_chat_message(request: ChatRequest):
    sender_id = request.sender_id
    user_message_text = request.message
    
    # recupera o histórico do MongoDB
    stored_messages = conversations_collection.find({"sender_id": sender_id}).sort("timestamp", 1)
    
    # reconstrói a lista de mensagens do LangChain a partir dos dados do banco
    current_history = messages_from_dict([msg["message_data"] for msg in stored_messages])
    
    # se for a primeira mensagem, envia a saudação inicial
    if not current_history:
        initial_message = AIMessage(content="Olá! Sou a assistente virtual da ClinicAI. Meu objetivo é coletar algumas informações para agilizar sua consulta. É importante lembrar que não substituo a avaliação de um profissional de saúde. Qual o motivo do seu contato hoje?")
        # salva a mensagem inicial no banco de dados
        conversations_collection.insert_one({
            "sender_id": sender_id,
            "message_data": message_to_dict(initial_message),
            "timestamp": datetime.datetime.now()
        })
        return {"response": initial_message.content}

    # adiciona a nova mensagem do usuário
    user_message = HumanMessage(content=user_message_text)
    current_history.append(user_message)
    
    # invoca o agente
    result = agent_app.invoke({"messages": current_history})
    agent_response = result['messages'][-1]
    
    # salva a mensagem do usuário E a resposta do agente no banco
    conversations_collection.insert_many([
        {"sender_id": sender_id, "message_data": message_to_dict(user_message), "timestamp": datetime.datetime.now()},
        {"sender_id": sender_id, "message_data": message_to_dict(agent_response), "timestamp": datetime.datetime.now()}
    ])
    
    return {"response": agent_response.content}