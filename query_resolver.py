import pandas as pd
from sentence_transformers import SentenceTransformer, InputExample, losses, util
from torch.utils.data import DataLoader

fine_tuned_model = SentenceTransformer("extracted_directory/contextual_query resolver")

def qa(fine_tuned_model, context, question, top_k=3):
    # Split context into sentences
    sentences = context.split(". ")

    # Get embeddings for the question and context sentences
    question_embedding = fine_tuned_model.encode(question, convert_to_tensor=True)
    sentence_embeddings = fine_tuned_model.encode(sentences, convert_to_tensor=True)

    # Calculate similarity scores between the question and all context sentences
    similarity_scores = util.pytorch_cos_sim(question_embedding, sentence_embeddings).flatten()

    # Get the top k most similar sentences
    top_k_indices = similarity_scores.topk(top_k).indices.tolist()

    # Combine the top-k sentences to form a more informative answer
    top_sentences = [sentences[idx] for idx in top_k_indices]
    answer = list((top_sentences))

    return answer


while True:
    # Ask for the context
    context = input("\n\nInput Context: (or type exit to leave) :\n\n")
    
    # Exit if user types 'exit'
    if context.lower() == 'exit':
        print("Nice to help you!")
        break
    
    print("\nContext has been set. You can now ask questions related to it.\n")
    
    # Ask questions related to the same context
    while True:
        question = input("\nEnter Question (or type 'change' to input a new context): ")
        
        # Exit the question loop if the user wants to input a new context
        if question.lower() == 'change':
            print("\nYou can now provide a new context.")
            break
        
        # Generate and display the answer to the current question
        gen_answer = qa(fine_tuned_model, context, question)
        print("\nGenerated Answer: ", gen_answer)