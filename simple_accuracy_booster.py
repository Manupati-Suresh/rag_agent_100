#!/usr/bin/env python3
"""
Simple and Effective Accuracy Booster
Focused approach to achieve 90% accuracy with proven techniques
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, List
from rag_agent import RAGAgent

class SimpleAccuracyBooster:
    """Simple, effective accuracy improvement system"""
    
    def __init__(self):
        self.rag_agent = None
        self.target_accuracy = 0.90
        
    async def initialize_system(self):
        """Initialize RAG system with local storage for reliability"""
        print("🚀 Initializing Simple Accuracy Booster")
        print("=" * 50)
        
        # Use local RAG agent for reliability
        self.rag_agent = RAGAgent()
        
        # Load existing documents
        self.rag_agent.load_documents()
        
        # Ensure we have documents
        doc_count = len(self.rag_agent.document_store.documents)
        print(f"📚 Documents loaded: {doc_count}")
        
        if doc_count == 0:
            print("⚠️ No documents found. Adding sample ESG documents...")
            self._add_sample_esg_documents()
        
        # Initialize search index
        self.rag_agent.initialize()
        
        print("✅ System initialized successfully!")
        
    def _add_sample_esg_documents(self):
        """Add focused ESG documents for testing"""
        esg_docs = [
            {
                'id': 'esg_basics',
                'content': '''ESG stands for Environmental, Social, and Governance. These are three key factors used to evaluate companies' sustainability and ethical impact. Environmental factors include climate change, carbon emissions, energy efficiency, waste management, and resource conservation. Social factors cover employee relations, diversity and inclusion, human rights, community engagement, and customer satisfaction. Governance factors involve board composition, executive compensation, business ethics, transparency, and shareholder rights. ESG criteria help investors make responsible investment decisions and companies improve their long-term sustainability performance.''',
                'metadata': {'category': 'esg_fundamentals'}
            },
            {
                'id': 'esg_reporting',
                'content': '''ESG reporting involves companies disclosing their environmental, social, and governance performance to stakeholders. Key components include materiality assessment, performance metrics, targets and goals, governance structure, risk management, and stakeholder engagement. Companies use frameworks like GRI (Global Reporting Initiative), SASB (Sustainability Accounting Standards Board), and TCFD (Task Force on Climate-related Financial Disclosures). Effective ESG reporting should be accurate, complete, consistent, comparable, and relevant. It helps build trust with investors, customers, employees, and communities while demonstrating commitment to sustainable business practices.''',
                'metadata': {'category': 'esg_reporting'}
            },
            {
                'id': 'esg_implementation',
                'content': '''Implementing ESG programs requires systematic planning and execution. Companies should start with leadership commitment and board oversight. Key steps include conducting materiality assessments, setting clear goals and targets, establishing governance structures, integrating ESG into business processes, collecting and analyzing data, engaging stakeholders, and reporting progress. Success factors include dedicated resources, employee training, stakeholder engagement, regular monitoring, and continuous improvement. ESG implementation creates value through risk mitigation, operational efficiency, innovation, stakeholder trust, and access to capital.''',
                'metadata': {'category': 'esg_implementation'}
            }
        ]
        
        for doc in esg_docs:
            self.rag_agent.document_store.add_document(doc['id'], doc['content'], doc['metadata'])
        
        print(f"✅ Added {len(esg_docs)} ESG documents")
    
    def run_simple_accuracy_test(self) -> Dict:
        """Run a simple, focused accuracy test"""
        print("\n📊 Running Simple Accuracy Test")
        print("-" * 40)
        
        # Simple test questions focused on your documents
        test_questions = [
            {
                'question': 'What is ESG?',
                'expected_keywords': ['environmental', 'social', 'governance', 'sustainability'],
                'category': 'basic'
            },
            {
                'question': 'What are ESG disclosures?',
                'expected_keywords': ['reporting', 'disclosure', 'stakeholders', 'transparency'],
                'category': 'basic'
            },
            {
                'question': 'How do companies implement ESG programs?',
                'expected_keywords': ['implementation', 'planning', 'leadership', 'goals'],
                'category': 'intermediate'
            },
            {
                'question': 'What are the benefits of ESG programs?',
                'expected_keywords': ['benefits', 'value', 'risk', 'efficiency'],
                'category': 'intermediate'
            },
            {
                'question': 'What frameworks are used for ESG reporting?',
                'expected_keywords': ['GRI', 'SASB', 'TCFD', 'frameworks'],
                'category': 'advanced'
            }
        ]
        
        results = []
        total_score = 0
        
        for i, test in enumerate(test_questions, 1):
            print(f"\n   Question {i}: {test['question']}")
            
            # Get response
            response = self.rag_agent.chat_with_documents(test['question'])
            
            if response['success']:
                answer = response['response']
                
                # Calculate accuracy based on keyword presence
                found_keywords = 0
                for keyword in test['expected_keywords']:
                    if keyword.lower() in answer.lower():
                        found_keywords += 1
                
                accuracy = found_keywords / len(test['expected_keywords'])
                total_score += accuracy
                
                print(f"      Answer: {answer[:100]}...")
                print(f"      Accuracy: {accuracy:.2%}")
                
                results.append({
                    'question': test['question'],
                    'answer': answer,
                    'accuracy': accuracy,
                    'category': test['category']
                })
            else:
                print(f"      ❌ Failed: {response.get('error', 'Unknown error')}")
                results.append({
                    'question': test['question'],
                    'answer': 'ERROR',
                    'accuracy': 0.0,
                    'category': test['category']
                })
        
        overall_accuracy = total_score / len(test_questions)
        
        print(f"\n📈 Overall Accuracy: {overall_accuracy:.2%}")
        print(f"🎯 Target: {self.target_accuracy:.2%}")
        print(f"✅ Target Achieved: {'YES' if overall_accuracy >= self.target_accuracy else 'NO'}")
        
        return {
            'overall_accuracy': overall_accuracy,
            'target_achieved': overall_accuracy >= self.target_accuracy,
            'detailed_results': results,
            'test_count': len(test_questions)
        }
    
    def apply_simple_optimizations(self):
        """Apply simple, proven optimizations"""
        print("\n🔧 Applying Simple Optimizations")
        print("-" * 40)
        
        # Optimization 1: Improve search relevance
        print("1. Improving search relevance...")
        original_search = self.rag_agent.search_documents
        
        def enhanced_search(query: str, top_k: int = 5):
            # Add ESG-specific terms to improve search
            esg_terms = {
                'esg': 'environmental social governance sustainability',
                'environmental': 'climate carbon emissions energy waste',
                'social': 'diversity inclusion employees community',
                'governance': 'board oversight compliance ethics'
            }
            
            enhanced_query = query
            for term, expansion in esg_terms.items():
                if term in query.lower():
                    enhanced_query += f" {expansion}"
            
            return original_search(enhanced_query, top_k)
        
        self.rag_agent.search_documents = enhanced_search
        print("   ✅ Search enhancement applied")
        
        # Optimization 2: Improve response generation
        print("2. Improving response generation...")
        original_chat = self.rag_agent.chat_with_documents
        
        def enhanced_chat(user_message: str, max_context_docs: int = 3):
            # Get enhanced search results
            relevant_docs = self.rag_agent.search_documents(user_message, top_k=max_context_docs)
            
            if not relevant_docs:
                return {
                    'success': True,
                    'response': "I don't have relevant information to answer your question about ESG topics.",
                    'sources': []
                }
            
            # Create focused prompt
            context = "\n\n".join([
                f"Document {i+1}: {doc['content']}"
                for i, doc in enumerate(relevant_docs)
            ])
            
            prompt = f"""Based on the following ESG documents, provide a comprehensive and accurate answer to the question.

ESG Documents:
{context}

Question: {user_message}

Instructions:
1. Use information directly from the provided documents
2. Focus on ESG (Environmental, Social, Governance) aspects
3. Provide specific details and examples when available
4. Be accurate and comprehensive
5. If the documents don't contain enough information, state this clearly

Answer:"""
            
            try:
                if self.rag_agent.gemini_model:
                    response = self.rag_agent.gemini_model.generate_content(prompt)
                    return {
                        'success': True,
                        'response': response.text,
                        'sources': [{'document_id': doc['document_id']} for doc in relevant_docs]
                    }
                else:
                    return original_chat(user_message, max_context_docs)
            except Exception as e:
                return {
                    'success': False,
                    'error': str(e),
                    'response': f'Error generating response: {str(e)}'
                }
        
        self.rag_agent.chat_with_documents = enhanced_chat
        print("   ✅ Response enhancement applied")
        
        print("✅ Simple optimizations completed")
    
    async def boost_accuracy(self) -> Dict:
        """Main accuracy boosting process"""
        print("\n🎯 SIMPLE ACCURACY BOOSTING PROCESS")
        print("=" * 50)
        
        # Step 1: Initial test
        print("📊 Step 1: Initial Accuracy Test")
        initial_results = self.run_simple_accuracy_test()
        initial_accuracy = initial_results['overall_accuracy']
        
        # Step 2: Apply optimizations if needed
        if initial_accuracy < self.target_accuracy:
            print(f"\n🔧 Step 2: Applying Optimizations (Current: {initial_accuracy:.2%} < Target: {self.target_accuracy:.2%})")
            self.apply_simple_optimizations()
            
            # Step 3: Re-test
            print("\n📊 Step 3: Post-Optimization Test")
            final_results = self.run_simple_accuracy_test()
            final_accuracy = final_results['overall_accuracy']
            
            improvement = final_accuracy - initial_accuracy
            
            return {
                'initial_accuracy': initial_accuracy,
                'final_accuracy': final_accuracy,
                'improvement': improvement,
                'target_achieved': final_accuracy >= self.target_accuracy,
                'optimization_applied': True,
                'initial_results': initial_results,
                'final_results': final_results
            }
        else:
            return {
                'initial_accuracy': initial_accuracy,
                'final_accuracy': initial_accuracy,
                'improvement': 0.0,
                'target_achieved': True,
                'optimization_applied': False,
                'initial_results': initial_results
            }
    
    def save_results(self, results: Dict):
        """Save results to file"""
        filename = f"simple_accuracy_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Make results JSON serializable
        serializable_results = self._make_json_serializable(results)
        
        with open(filename, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"\n💾 Results saved to: {filename}")
    
    def _make_json_serializable(self, obj):
        """Convert objects to JSON-serializable format"""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, (bool, int, float, str)):
            return obj
        elif obj is None:
            return obj
        else:
            return str(obj)


async def main():
    """Main function"""
    print("🎯 Simple Accuracy Booster for RAG Bot")
    print("Focused approach to achieve 90% accuracy")
    print("=" * 50)
    
    # Create booster
    booster = SimpleAccuracyBooster()
    
    # Initialize
    await booster.initialize_system()
    
    # Boost accuracy
    results = await booster.boost_accuracy()
    
    # Display results
    print("\n🎉 ACCURACY BOOSTING RESULTS")
    print("=" * 50)
    print(f"Initial Accuracy: {results['initial_accuracy']:.2%}")
    print(f"Final Accuracy: {results['final_accuracy']:.2%}")
    print(f"Improvement: {results['improvement']:.2%}")
    print(f"Target Achieved: {'✅ YES' if results['target_achieved'] else '❌ NO'}")
    
    if results['target_achieved']:
        print(f"\n🎊 CONGRATULATIONS! 🎊")
        print(f"Your RAG bot achieved {results['final_accuracy']:.2%} accuracy!")
    else:
        print(f"\n💡 Recommendations:")
        print(f"   📚 Add more comprehensive documents covering your specific topics")
        print(f"   🔧 Consider domain-specific optimizations")
        print(f"   📊 Analyze failed questions for patterns")
    
    # Save results
    booster.save_results(results)
    
    print(f"\n✅ Simple accuracy boosting completed!")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print(f"\n❌ Process interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print(f"Please ensure your system is properly configured")