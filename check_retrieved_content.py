# save as check_retrieved_content.py
import os
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone

load_dotenv()

# Initialize clients
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))

# Function to test a query
def test_query(query_text, top_k=5):
    print(f"\n{'='*80}\nTESTING QUERY: {query_text}\n{'='*80}")
    
    # Get embedding
    response = openai_client.embeddings.create(
        input=query_text,
        model="text-embedding-ada-002"
    )
    query_vector = response.data[0].embedding
    
    # Query Pinecone
    results = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True
    )
    
    # Print results
    print(f"Found {len(results['matches'])} matches:")
    for i, match in enumerate(results['matches']):
        print(f"\n--- Match {i+1} (Score: {match['score']:.4f}) ---")
        # Print source if available
        if 'source' in match['metadata']:
            print(f"Source: {match['metadata']['source']}")
        if 'chapter' in match['metadata']:
            print(f"Chapter: {match['metadata']['chapter']}")
        if 'chunk_type' in match['metadata']:
            print(f"Type: {match['metadata']['chunk_type']}")
            
        # Print preview of text
        text = match['metadata']['text']
        print(text[:300] + "..." if len(text) > 300 else text)

# Get index stats
stats = index.describe_index_stats()
print(f"Total vectors in index: {stats.total_vector_count}")

# Test the problematic query that was failing before
test_query("care sunt sesiunile de tranzactionare")

# Test some other important queries
test_query("ce este un mss agresiv")
test_query("cum se calculeaza stop loss")
test_query("care sunt cele mai recomandate carti de trading")