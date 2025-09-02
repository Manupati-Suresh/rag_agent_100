#!/usr/bin/env python3
"""
Setup script to download required NLTK data
"""

def setup_nltk():
    """Download required NLTK data"""
    try:
        import nltk
        print("📦 Setting up NLTK data...")
        
        # List of required NLTK resources
        resources = [
            'punkt',
            'punkt_tab', 
            'stopwords',
            'wordnet',
            'averaged_perceptron_tagger',
            'averaged_perceptron_tagger_eng',
            'omw-1.4'  # For wordnet
        ]
        
        successful_downloads = []
        failed_downloads = []
        
        for resource in resources:
            try:
                print(f"  Downloading {resource}...")
                nltk.download(resource, quiet=True)
                successful_downloads.append(resource)
                print(f"  ✅ {resource} downloaded successfully")
            except Exception as e:
                failed_downloads.append((resource, str(e)))
                print(f"  ⚠️ {resource} failed: {str(e)}")
        
        print(f"\n📊 Download Summary:")
        print(f"  ✅ Successful: {len(successful_downloads)}")
        print(f"  ⚠️ Failed: {len(failed_downloads)}")
        
        if successful_downloads:
            print(f"  Successfully downloaded: {', '.join(successful_downloads)}")
        
        if failed_downloads:
            print(f"  Failed downloads:")
            for resource, error in failed_downloads:
                print(f"    - {resource}: {error}")
        
        # Test basic functionality
        print(f"\n🧪 Testing NLTK functionality...")
        
        try:
            from nltk.tokenize import word_tokenize, sent_tokenize
            from nltk.corpus import stopwords
            from nltk.stem import WordNetLemmatizer
            
            # Test tokenization
            test_text = "This is a test sentence. Here is another one!"
            words = word_tokenize(test_text)
            sentences = sent_tokenize(test_text)
            
            print(f"  ✅ Tokenization works: {len(words)} words, {len(sentences)} sentences")
            
            # Test stopwords
            stops = stopwords.words('english')
            print(f"  ✅ Stopwords loaded: {len(stops)} English stopwords")
            
            # Test lemmatizer
            lemmatizer = WordNetLemmatizer()
            lemma = lemmatizer.lemmatize('running', 'v')
            print(f"  ✅ Lemmatization works: 'running' -> '{lemma}'")
            
            print(f"\n🎉 NLTK setup completed successfully!")
            return True
            
        except Exception as e:
            print(f"  ❌ NLTK functionality test failed: {str(e)}")
            print(f"  ℹ️ The system will fall back to basic text processing")
            return False
            
    except ImportError:
        print("❌ NLTK not installed. Please install it with: pip install nltk")
        return False
    except Exception as e:
        print(f"❌ NLTK setup failed: {str(e)}")
        return False

if __name__ == "__main__":
    setup_nltk()