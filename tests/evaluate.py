from ragas.metrics import Faithfulness, AnswerRelevancy
from ragas import evaluate
from datasets import Dataset
from src.retrieval.rag import query_rag
from src.retrieval.retriever import retrieve, get_collection
from langchain_openai import OpenAIEmbeddings
import json

# Fix Answer Relevancy NaN issue
answer_relevancy = AnswerRelevancy()
answer_relevancy.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

test_questions = [
    # Transformers / Attention
    "What is the attention mechanism in transformers?",
    "How does multi-head attention work?",
    "What is self-attention?",
    
    # RAG
    "What is retrieval augmented generation?",
    "How does RAG combine retrieval with generation?",
    "What are the benefits of RAG over fine-tuning?",
    
    # LoRA
    "How does LoRA reduce training parameters?",
    "What are low-rank matrices in LoRA?",
    "How does LoRA work during inference?",
    
    # InstructGPT
    "How was InstructGPT trained using human feedback?",
    "What is RLHF and how does it work?",
    "What is a reward model in RLHF?",
    
    # Chain of Thought
    "What is chain of thought prompting?",
    "How does chain of thought improve reasoning?",
    "What are few-shot examples in chain of thought?",
    
    # ReAct
    "What is the ReAct framework?",
    "How does ReAct combine reasoning and acting?",
    
    # LLaMA
    "What is LLaMA and how was it trained?",
    "How does LLaMA 2 differ from LLaMA?",
    
    # GPT-4
    "What are the key capabilities of GPT-4?",
    "How does GPT-4 handle multimodal inputs?",
    
    # LLM Survey
    "What are scaling laws in large language models?",
    "What are emergent abilities in LLMs?",
    
    # Cross-paper
    "How does RLHF help align language models?",
    "What is the difference between LoRA and full fine-tuning?",
]

def run_evaluation():
    print(f"Running RAG pipeline on {len(test_questions)} test questions...\n")
    
    questions = []
    answers = []
    contexts = []
    
    collection = get_collection()
    
    for i, question in enumerate(test_questions):
        print(f"[{i+1}/{len(test_questions)}] {question}")
        result = query_rag(question)
        chunks = retrieve(question, collection, top_k=5)
        context = [c["content"] for c in chunks]
        questions.append(question)
        answers.append(result["answer"])
        contexts.append(context)
    
    print("\nAll questions processed!")
    print("Running RAGAs evaluation...\n")
    
    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
    })
    
    metrics = [Faithfulness(), answer_relevancy]
    result = evaluate(dataset, metrics=metrics)
    
    df = result.to_pandas()
    
    faith_score = df['faithfulness'].mean()
    relevancy_score = df['answer_relevancy'].mean()
    
    print("\nEVALUATION RESULTS:")
    print("=" * 40)
    print(f"Faithfulness:     {faith_score:.2%}")
    print(f"Answer Relevancy: {relevancy_score:.2%}")
    print(f"Test Questions:   {len(test_questions)}")
    print("=" * 40)
    
    scores = {
        "faithfulness": float(faith_score),
        "answer_relevancy": float(relevancy_score),
        "num_questions": len(test_questions)
    }
    
    with open("data/processed/eval_results.json", "w") as f:
        json.dump(scores, f, indent=2)
    
    print("\nResults saved! ✅")
    return scores

if __name__ == "__main__":
    run_evaluation()