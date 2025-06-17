import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX_NAME", "trading-lessons"))

def diagnose_retrieval_system():
    """Comprehensive diagnosis of the retrieval system."""
    print("🔍 DIAGNOSTIC ANALYSIS OF RETRIEVAL SYSTEM")
    print("=" * 50)
    
    # 1. Check index stats
    print("\n1. 📊 INDEX STATISTICS")
    try:
        stats = index.describe_index_stats()
        print(f"   Total vectors: {stats.total_vector_count}")
        print(f"   Dimension: {stats.dimension}")
        print(f"   Namespaces: {list(stats.namespaces.keys()) if stats.namespaces else 'None'}")
    except Exception as e:
        print(f"   ❌ Error getting stats: {e}")
    
    # 2. Test different embedding models
    print("\n2. 🧠 EMBEDDING MODEL TESTS")
    test_queries = [
        "cum identific un mss agresiv?",
        "ce sunt tipurile de lichiditate?", 
        "cum fac trade management?",
        "aggressive MSS identification",
        "liquidity types trading"
    ]
    
    embedding_models = [
        "text-embedding-3-small",
        "text-embedding-ada-002",
        "text-embedding-3-large"
    ]
    
    for model in embedding_models:
        print(f"\n   Testing model: {model}")
        try:
            for query in test_queries[:2]:  # Test first 2 queries
                embedding = get_embedding_with_model(query, model)
                if embedding:
                    results = index.query(vector=embedding, top_k=3, include_metadata=True)
                    if results['matches']:
                        top_score = results['matches'][0].score
                        print(f"     '{query}' -> Top score: {top_score:.4f}")
                    else:
                        print(f"     '{query}' -> No results")
        except Exception as e:
            print(f"     ❌ Error with {model}: {e}")
    
    # 3. Sample vector content analysis
    print("\n3. 📄 VECTOR CONTENT ANALYSIS")
    try:
        # Get a random sample of vectors
        sample_results = index.query(
            vector=[0.1] * 1536,  # Random vector to get any results
            top_k=5,
            include_metadata=True
        )
        
        print(f"   Sample vectors found: {len(sample_results['matches'])}")
        for i, match in enumerate(sample_results['matches'][:3]):
            if match.metadata:
                text_preview = match.metadata.get('text', 'No text')[:100]
                chapter = match.metadata.get('chapter', 'Unknown')
                print(f"     Sample {i+1}: Chapter {chapter}")
                print(f"       Text: {text_preview}...")
                print(f"       Score: {match.score:.4f}")
    except Exception as e:
        print(f"   ❌ Error sampling content: {e}")
    
    # 4. Direct content search
    print("\n4. 🔍 DIRECT CONTENT SEARCH")
    search_terms = ["mss agresiv", "lichiditate", "setup", "gap", "pivot"]
    
    try:
        # Get more vectors to search through
        broad_results = index.query(
            vector=[0.0] * 1536,  # Zero vector
            top_k=50,
            include_metadata=True
        )
        
        for term in search_terms:
            found_count = 0
            for match in broad_results['matches']:
                if match.metadata and 'text' in match.metadata:
                    if term.lower() in match.metadata['text'].lower():
                        found_count += 1
            print(f"   '{term}' found in {found_count} vectors")
    except Exception as e:
        print(f"   ❌ Error in direct search: {e}")
    
    # 5. Embedding dimension check
    print("\n5. 🔢 EMBEDDING DIMENSION CHECK")
    try:
        test_embedding = get_embedding_with_model("test", "text-embedding-3-small")
        if test_embedding:
            print(f"   Current embedding dimension: {len(test_embedding)}")
            print(f"   Index dimension: {stats.dimension if 'stats' in locals() else 'Unknown'}")
            if len(test_embedding) != stats.dimension:
                print("   ⚠️  DIMENSION MISMATCH DETECTED!")
    except Exception as e:
        print(f"   ❌ Error checking dimensions: {e}")
    
    # 6. Score distribution analysis
    print("\n6. 📈 SCORE DISTRIBUTION ANALYSIS")
    try:
        query = "cum identific un mss agresiv?"
        embedding = get_embedding_with_model(query, "text-embedding-3-small")
        if embedding:
            results = index.query(vector=embedding, top_k=20, include_metadata=True)
            scores = [match.score for match in results['matches']]
            if scores:
                print(f"   Query: '{query}'")
                print(f"   Score range: {min(scores):.4f} - {max(scores):.4f}")
                print(f"   Average score: {sum(scores)/len(scores):.4f}")
                print(f"   Scores: {[f'{s:.3f}' for s in scores[:10]]}")
    except Exception as e:
        print(f"   ❌ Error in score analysis: {e}")

def get_embedding_with_model(text, model):
    """Get embedding with specific model."""
    try:
        response = client.embeddings.create(model=model, input=text)
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Error getting embedding with {model}: {e}")
        return None

def recommend_fixes():
    """Generate specific recommendations based on diagnosis."""
    print("\n" + "=" * 50)
    print("🔧 RECOMMENDED FIXES")
    print("=" * 50)
    
    print("""
1. IMMEDIATE FIXES:
   - Lower thresholds to 0.30-0.35 range
   - Check embedding model consistency
   - Test with different embedding models

2. CONTENT VERIFICATION:
   - Verify MSS content exists in vector DB
   - Check Romanian language handling
   - Review chunking strategy

3. SYSTEM IMPROVEMENTS:
   - Implement fallback mechanisms
   - Add query preprocessing
   - Consider re-embedding with consistent model

4. MONITORING:
   - Add score distribution logging
   - Monitor threshold effectiveness
   - Track query success rates
""")

if __name__ == "__main__":
    diagnose_retrieval_system()
    recommend_fixes() 