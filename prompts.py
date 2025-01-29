import streamlit as st
from examples import get_example_selector
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder,FewShotChatMessagePromptTemplate,PromptTemplate


condense_question_prompt = PromptTemplate(
    input_variables=['chat_history', 'question'], 
    template="""Given the following conversation and a follow up question (at the end), 
rephrase the follow up question to be a standalone question, in the same language as the follow up question.\n\n
Chat History:\n{chat_history}\n
Keep in mind that the last human message in chat history is the same as the follow up question.\n
Follow up question: {question}\n
Standalone question:"""
)

def answer_template(language="english"):
    """Pass the standalone question along with the chat history and context 
    to the `LLM` wihch will answer"""
    
    template = f"""Answer the question at the end, using only the following context (delimited by <context></context>).
Your answer must be in the language at the end. 

<context>
{{chat_history}}

{{context}} 
</context>

Question: {{question}}

Language: {language}.
"""
    return template
