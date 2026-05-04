from ragas.metrics import Faithfulness, AnswerRelevancy
from ragas import evaluate
from datasets import Dataset
from src.retrieval.rag import query_rag
from src.retrieval.retriever import retrieve, get_collection
import json

test_questions = [
    "What is the attention mechanism in transformers?",
    "What is LoRA and how does it work?",
    "What is chain of thought prompting?",
    "What is RAG and why is it useful?",
    "What is InstructGPT and how was it trained?",
]

def run_evaluation():
    print("Running RAG pipeline on test questions...\n")
    
    questions = []
    answers = []
    contexts = []
    
    collection = get_collection()
    
    for question in test_questions:
        print(f"Processing: {question}")
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
    
    metrics = [Faithfulness(), AnswerRelevancy()]
    result = evaluate(dataset, metrics=metrics)
    
    # Convert result to dict and print everything
    result_dict = result.to_pandas().to_dict()
    print("\nRAW RESULTS:")
    print(result_dict)
    
    # Get scores safely
    df = result.to_pandas()
    print("\nEVALUATION RESULTS:")
    print("=" * 40)
    print(df[['faithfulness', 'answer_relevancy']].mean())
    print("=" * 40)
    
    # Save
    scores = df[['faithfulness', 'answer_relevancy']].mean().to_dict()
    with open("data/processed/eval_results.json", "w") as f:
        json.dump(scores, f, indent=2)
    
    print("\nResults saved! ✅")
    return scores

if __name__ == "__main__":
    run_evaluation()