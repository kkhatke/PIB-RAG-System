"""
Setup verification script for PIB RAG System
Checks that all required dependencies are installed and accessible
"""

import sys

def check_imports():
    """Check if all required packages can be imported"""
    required_packages = {
        'langchain': 'LangChain',
        'langchain_community': 'LangChain Community',
        'chromadb': 'ChromaDB',
        'sentence_transformers': 'Sentence Transformers',
        'ollama': 'Ollama',
        'hypothesis': 'Hypothesis',
        'pytest': 'Pytest',
        'dotenv': 'Python-dotenv'
    }
    
    print("Checking required packages...\n")
    all_ok = True
    
    for package, name in required_packages.items():
        try:
            __import__(package)
            print(f"✓ {name} - OK")
        except ImportError as e:
            print(f"✗ {name} - FAILED: {e}")
            all_ok = False
    
    return all_ok

def check_config():
    """Check if configuration file exists and is valid"""
    print("\n\nChecking configuration...")
    try:
        import config
        print(f"✓ Configuration file loaded")
        print(f"  - Embedding Model: {config.EMBEDDING_MODEL}")
        print(f"  - Ollama Base URL: {config.OLLAMA_BASE_URL}")
        print(f"  - Ollama Model: {config.OLLAMA_MODEL}")
        print(f"  - Vector Store Directory: {config.VECTOR_STORE_DIR}")
        return True
    except Exception as e:
        print(f"✗ Configuration check failed: {e}")
        return False

def check_directories():
    """Check if required directories exist"""
    print("\n\nChecking project structure...")
    import os
    
    required_dirs = [
        'src/data_ingestion',
        'src/embedding',
        'src/vector_store',
        'src/query_engine',
        'src/response_generation',
        'src/interface'
    ]
    
    all_ok = True
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"✓ {dir_path} - OK")
        else:
            print(f"✗ {dir_path} - MISSING")
            all_ok = False
    
    return all_ok

def main():
    """Run all verification checks"""
    print("=" * 60)
    print("PIB RAG System - Setup Verification")
    print("=" * 60)
    
    checks = [
        ("Package imports", check_imports),
        ("Configuration", check_config),
        ("Directory structure", check_directories)
    ]
    
    results = []
    for name, check_func in checks:
        results.append(check_func())
    
    print("\n" + "=" * 60)
    if all(results):
        print("✓ All checks passed! Setup is complete.")
        print("\nNext steps:")
        print("1. Ensure Ollama is installed and running")
        print("2. Pull the required model: ollama pull llama3.2")
        print("3. Run the data ingestion script to load articles")
        print("4. Start the conversational interface")
        return 0
    else:
        print("✗ Some checks failed. Please review the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
