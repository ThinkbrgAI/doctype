from config import SecureConfig, setup_api_key

def test_config():
    print("Testing configuration system...")
    
    # Try to set up API key
    print("\nAttempting to set up API key...")
    api_key = setup_api_key()
    
    if api_key:
        print("\nAPI key was successfully configured!")
        
        # Verify we can read it back
        config = SecureConfig()
        retrieved_key = config.get_api_key()
        
        if retrieved_key == api_key:
            print("✓ Successfully verified key storage and retrieval")
        else:
            print("✗ Error: Retrieved key doesn't match stored key")
    else:
        print("\n✗ Failed to set up API key")

if __name__ == "__main__":
    test_config() 