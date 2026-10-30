from pathlib import Path
from langchain_ollama import ChatOllama
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import ConversationalRetrievalChain
import pandas as pd
from ragas import EvaluationDataset
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import LLMContextRecall,Faithfulness,FactualCorrectness
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu
from transformers import BartForConditionalGeneration,BartTokenizer
from bert_score import BERTScorer
import torch
# Reading file and splitiing them
pdf_path = Path(r"C:\Users\HP\Desktop\New ML\Evaluation\Power+BI+Ebook.pdf")
pdf_reader = PyPDFLoader(str(pdf_path))
documents = pdf_reader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
chunks = text_splitter.split_documents(documents)
# Create embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.from_documents(documents=chunks,embedding=embeddings)
# Calling the model
llm = ChatOllama(model="gemma3:1b",temperature=0)
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(
    """Given the following conversation and a follow-up question,
rephrase the follow-up question into a standalone question.
Chat History:
{chat_history}
Follow-up Input:
{question}
Standalone Question:""")
# Build the conversational RAG chain
qa = ConversationalRetrievalChain.from_llm(llm=llm,retriever=db.as_retriever(),condense_question_prompt=CONDENSE_QUESTION_PROMPT,return_source_documents=True,verbose=False)
chat_history = []
query = """What are the components of Power BI? """
result = qa({'question' : query,'chat_history' : chat_history})
print(result['answer'])
# RAG EVALUATION
sample_queries = [
    {"question" : "What are the main components of Power BI?",
     "answer" : "The main components of Power BI are Power BI desktop,Power Bi service and Power BI Mobile",
     "reference" : "Power BI has three components: Desktop(Window app for creating reports),Service(cloud platform for sharing and collaboration),Mobile(app for viewing reports on mobile device)"
     },
    {"question" : "What are the key features of Power BI Desktop?",
     "answer" : "Key features include data connection and transformation,interactive visualisations,report publishing and collaboration tools",
     "reference" : "Power BI Desktop allows users to get data,analyze it with DAX,visualize using 150+ visuals,publish to cloud or on-premise and collaborate with team members"},
    {"question" : "What is the Power Query Editor used for?",
     "answer" : "Power Query Editor is used for data transformation and cleansing before importing data into models or visualizations.",
     "reference" : "Power Query Editor includes tools like ribbon tabs,data preview,query settings,and footer with data stats for transforming and shaping data"},
    {"question" : "How can you group data into Power BI?",
     "answer" : "You can group data using the 'Group By' feature in the transform tab of the Query Editor by selecting columns and specifying aggregate calculations",
     "reference" : "In Power BI's Query Editor,the Group By dialog allows users to select columns and apply aggregate functions to group data"}
]

df = pd.DataFrame(sample_queries)
retriever = db.as_retriever()
def process_query(query):
    chat_history = []
    result = qa({'question':query,'chat_history':chat_history})
    relevant_docs = retriever.invoke(query)
    print(result['answer'])
    return result['answer'],relevant_docs
process_query('What is the Power Query editor used for?')

# RAGAS Framework
results = []
for _,row in df.iterrows():
    question = row['question']
    reference = row['reference']
    answer,relevant_docs = process_query(question)
    results.append({
        'user_input' : question,
        'reference' : reference,
        'response' : answer,
        'retrieved_contexts' : [relevant_docs[0].page_content]
    })
    
evaluation_dataset = EvaluationDataset.from_list(results)
evaluator_llm = LangchainLLMWrapper(llm)
ragas_results = evaluate(
    dataset = evaluation_dataset,
    metrics = [LLMContextRecall(),Faithfulness(),FactualCorrectness()],
    llm = evaluator_llm
)
print(ragas_results)
# Rouge and bleu score
rouge = Rouge()
bleu = []
rouge_one = []
for item in results:
    reference = [item["reference"].split()]
    hypothesis = item["response"].split()
    bleu.append(sentence_bleu(reference,hypothesis))
    scores = rouge.get_scores(item["response"],item["reference"])[0]
    rouge_one.append(scores['rouge-1']['f'])
print(bleu)
print(rouge_one)
# Initialize the BART model and tokenizer
bart_model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
bart_tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
# Initialise BERT Score
bert_scorer = BERTScorer(lang='en',rescale_with_baseline=True)
bert_scores = []
bart_scores = []
for item in results:
    inputs = bart_tokenizer(item["response"],truncation=True,padding=True,return_tensors='pt')
    with torch.no_grad():
        bart_score = bart_model(**inputs).logits
    bart_scores.append(bart_score.mean().item())
    P,R,F1 = bert_scorer.score([item["response"]],[item["reference"]])
    bert_scores.append(F1.numpy().mean())
print(bart_scores)
print(bert_scores)