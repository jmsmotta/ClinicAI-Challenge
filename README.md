# 🤖 ClinicAI - Agente de Triagem com IA Generativa

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110-green?style=for-the-badge&logo=fastapi)
![LangGraph](https://img.shields.io/badge/LangGraph-AI-orange?style=for-the-badge)
![MongoDB](https://img.shields.io/badge/MongoDB-Atlas-darkgreen?style=for-the-badge&logo=mongodb)

## 📄 Descrição do Projeto

**ClinicAI** é a solução para um desafio técnico que consiste em criar um agente de IA para realizar triagens clínicas iniciais. O agente é projetado para conversar com pacientes, coletar informações essenciais sobre seus sintomas de forma empática e estruturada, e identificar casos de emergência, fornecendo orientação imediata.

O coração do projeto é um backend robusto construído em Python, utilizando o **FastAPI** para a criação da API, **LangGraph** para orquestrar o fluxo da conversa e a lógica de estados, **Google Gemini** como o modelo de linguagem generativa, e **MongoDB Atlas** para garantir a persistência da memória e do histórico das conversas.

## ✨ Funcionalidades Principais

- **Agente Conversacional Inteligente:** Utiliza o Google Gemini para manter conversas fluidas e naturais.
- **Persona Definida:** O agente atua com uma persona acolhedora, empática e profissional, guiando o paciente de forma segura.
- **Coleta Estruturada de Dados:** Segue um roteiro para obter informações cruciais como queixa principal, sintomas, duração, intensidade, etc.
- **Protocolo de Emergência Determinístico:** Possui uma camada de segurança que identifica palavras-chave de emergência e aciona um protocolo imediato, sem depender do LLM, garantindo 100% de confiabilidade.
- **Memória Persistente:** Todas as interações são salvas no MongoDB Atlas, permitindo que o agente se lembre do histórico da conversa mesmo após o reinício do servidor.
- **Interface Web para Demonstração:** Um protótipo de frontend em HTML, CSS e JavaScript que permite a interação direta com o agente.

## ⚙️ Arquitetura e Tecnologias

| Componente      | Tecnologia Utilizada                               |
| --------------- | -------------------------------------------------- |
| **Backend** | Python 3.10, FastAPI, Uvicorn                      |
| **IA & Orquestração** | Google Gemini 2.0 Flash, LangGraph, LangChain    |
| **Banco de Dados** | MongoDB Atlas (Cloud)                              |
| **Frontend** | HTML5, CSS3, JavaScript (Fetch API)                |
| **Dependências** | `python-dotenv`, `pymongo`, `requests`             |

## 🎯 Contexto do Projeto

O objetivo inicial era integrar o agente a um chatbot no WhatsApp. No entanto, durante o desenvolvimento, foram encontrados bloqueios na API de desenvolvedor da Meta (restrições de conta em modo sandbox). Para garantir a entrega de uma solução 100% funcional e focar na robustez do backend, foi tomada a decisão estratégica de pivotar para um protótipo web. O arquivo `tentativaWPP.py` foi mantido no repositório como um artefato dessa fase inicial de desenvolvimento.

## 🚀 Como Executar o Projeto

Siga os passos abaixo para configurar e rodar o projeto em um ambiente local (recomenda-se WSL/Ubuntu).

### Pré-requisitos

- **Python 3.10+**
- **Git**
- **Conta no MongoDB Atlas:** [Crie uma conta gratuita aqui](https://www.mongodb.com/cloud/atlas).
- **Chave de API do Google AI Studio:** [Obtenha sua chave aqui](https://aistudio.google.com/).

### 1. Clonar o Repositório

```bash
git clone [URL_DO_SEU_REPOSITORIO_GIT]
cd [NOME_DA_PASTA_DO_REPOSITORIO]
```

### 2. Configurar o Ambiente Virtual

É uma boa prática usar um ambiente virtual para isolar as dependências do projeto.

```bash
# Criar o ambiente
python3 -m venv .venv

# Ativar o ambiente
source .venv/bin/activate
```

### 3. Instalar as Dependências

Crie um arquivo `requirements.txt` com o conteúdo abaixo ou use o que já está no repositório.

```txt
fastapi[all]
uvicorn[standard]
python-dotenv
pymongo[srv]
langchain-google-genai
langgraph
requests
```

Em seguida, instale todas as bibliotecas de uma vez:
```bash
pip install -r requirements.txt
```

### 4. Configurar as Variáveis de Ambiente

Crie um arquivo chamado `.env` na raiz do projeto, copie o conteúdo do exemplo abaixo e preencha com suas chaves.

```dotenv
# .env

# Chave de API obtida no Google AI Studio
GOOGLE_API_KEY="SUA_CHAVE_DE_API_DO_GOOGLE_AQUI"

# String de conexão obtida no MongoDB Atlas
# Lembre-se de substituir <password> pela senha do seu usuário do banco de dados
MONGO_URI="SUA_STRING_DE_CONEXAO_DO_MONGODB_ATLAS_AQUI"
```

### 5. Iniciar o Servidor

Com o ambiente virtual ativo, inicie a API FastAPI com o Uvicorn.

```bash
uvicorn ClinicAI:app --reload
```
O servidor estará rodando em `http://127.0.0.1:8000`.

## 💻 Como Usar

1.  Após iniciar o servidor, abra o arquivo `index.html` em seu navegador.
2.  A interface do chat aparecerá.
3.  Comece a conversar com o agente ClinicAI!

## 📂 Estrutura dos Arquivos

```
.
├── ClinicAI.py        # Arquivo principal do backend (FastAPI, LangGraph, MongoDB)
├── index.html         # Frontend do protótipo de chat
├── tentativaWPP.py    # Arquivo legado da tentativa de integração com o WhatsApp
├── requirements.txt   # Lista de dependências Python
└── README.md          # Este arquivo
```