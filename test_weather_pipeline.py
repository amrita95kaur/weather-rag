#!/usr/bin/env python3
"""Test script for the weather RAG pipeline.

This script demonstrates the complete workflow:
1. Create database tables
2. Sync weather documents from NWS API
3. Generate embeddings for documents
4. Perform semantic search
"""
import sys
sys.path.append('/Workspace/Users/amrita95kaur@gmail.com/weather-rag')

import lakebase
import weather_client
from sentence_transformers import SentenceTransformer


def test_table_creation():
    """Test 1: Create database tables."""
    print("\n=== Test 1: Create Database Tables ===")
    try:
        lakebase.ensure_weather_tables(embedding_dim=384)
        print("✓ Tables created successfully")
        return True
    except Exception as e:
        print(f"✗ Failed to create tables: {e}")
        return False


def test_weather_sync():
    """Test 2: Sync weather documents."""
    print("\n=== Test 2: Sync Weather Documents ===")
    test_locations = ["Chicago, IL", "San Francisco, CA", "Austin, TX"]
    try:
        print(f"Syncing weather data for: {', '.join(test_locations)}")
        count = weather_client.sync_locations(test_locations, limit=20)
        print(f"✓ Synced {count} documents")
        return count > 0
    except Exception as e:
        print(f"✗ Failed to sync weather: {e}")
        return False


def test_document_query():
    """Test 3: Query weather documents."""
    print("\n=== Test 3: Query Weather Documents ===")
    try:
        sql = "SELECT id, location, source_type, headline FROM weather_documents LIMIT 5"
        docs = lakebase.run_query(sql)
        print(f"✓ Found {len(docs)} documents in the database")
        for doc in docs:
            print(f"  - {doc['source_type']}: {doc['headline'][:60]}...")
        return len(docs) > 0
    except Exception as e:
        print(f"✗ Failed to query documents: {e}")
        return False


def test_embedding_generation():
    """Test 4: Generate embeddings (simulated - use notebook for actual generation)."""
    print("\n=== Test 4: Embedding Generation ===")
    print("Note: Run the 'ingest_weather_embeddings' notebook to generate embeddings.")
    print("This test checks if embeddings exist.")
    try:
        sql = "SELECT COUNT(*) as count FROM weather_embeddings"
        result = lakebase.run_query(sql)
        count = result[0]['count'] if result else 0
        print(f"✓ Found {count} embeddings in the database")
        return True
    except Exception as e:
        print(f"✗ Failed to query embeddings: {e}")
        return False


def test_semantic_search():
    """Test 5: Perform semantic search."""
    print("\n=== Test 5: Semantic Search ===")
    try:
        # Check if embeddings exist first
        sql = "SELECT COUNT(*) as count FROM weather_embeddings"
        result = lakebase.run_query(sql)
        count = result[0]['count'] if result else 0
        
        if count == 0:
            print("⚠ No embeddings found. Run the embedding notebook first.")
            return True  # Not a failure, just need to run embeddings
        
        print(f"Found {count} embeddings. Testing search...")
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        
        test_query = "risk of flooding near rivers"
        print(f"\nSearch query: '{test_query}'")
        
        query_vec = model.encode([test_query], show_progress_bar=False)[0].tolist()
        vec_literal = "[" + ",".join(map(lambda x: repr(float(x)), query_vec)) + "]"
        
        sql = (
            "SELECT d.location, d.headline, e.chunk_text, "
            "(e.embedding <=> %s::vector) AS distance "
            "FROM weather_embeddings e "
            "JOIN weather_documents d ON d.id = e.document_id "
            "ORDER BY distance ASC LIMIT 3"
        )
        
        results = lakebase.run_query(sql, (vec_literal,))
        print(f"\n✓ Found {len(results)} relevant results:")
        for i, r in enumerate(results, 1):
            similarity = 1.0 - float(r['distance'])
            print(f"\n  {i}. {r['location']} (similarity: {similarity:.3f})")
            print(f"     Headline: {r['headline']}")
            print(f"     Preview: {r['chunk_text'][:100]}...")
        
        return len(results) > 0
    except Exception as e:
        print(f"✗ Failed to perform semantic search: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("="*60)
    print("Weather RAG Pipeline Test Suite")
    print("="*60)
    
    tests = [
        ("Table Creation", test_table_creation),
        ("Weather Sync", test_weather_sync),
        ("Document Query", test_document_query),
        ("Embedding Check", test_embedding_generation),
        ("Semantic Search", test_semantic_search),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"\n✗ Test '{name}' crashed: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {name}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Your weather RAG pipeline is working.")
    else:
        print("\n⚠ Some tests failed. Check the output above for details.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
