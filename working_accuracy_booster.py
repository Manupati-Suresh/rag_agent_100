#!/usr/bin/env python3
"""
Working Accuracy Booster
Simple, effective accuracy improvement using the actual RAGAgent methods
"""

import json
from datetime import datetime
from typing import Dict, List
from rag_agent import RAGAgent
import re

class WorkingAccuracyBooster:
    """Working accuracy improvement system using actual RAGAgent methods"""
    
    def __init__(self):
        self.rag_agent = None
        self.target_accuracy = 0.90
        
    def initialize_system(self):
        """Initialize RAG system"""
        print("🚀 Initializing Working Accuracy Booster")
        print("=" * 50)
        
        # Use RAG agent
        self.rag_agent = RAGAgent()
        
        # Load documents (this will load existing or create samples)
        self.rag_agent.load_documents()
        
        # Initialize the agent (builds search index)
        self.rag_agent.initialize()
        
        doc_count = len(self.rag_agent.document_store.documents)
        print(f"📚 Documents loaded: {doc_count}")
        print("✅ System initialized successfully!")
        
    def local_answer_generation(self, question: str, relevant_docs: List[Dict]) -> str:
        """Generate answers using local processing"""
        if not relevant_docs:
            return "I don't have relevant information to answer your question."
        
        # Combine relevant document content
        combined_content = " ".join([doc['content'] for doc in relevant_docs[:3]])
        
        # Extract key sentences that might answer the question
        question_words = set(re.findall(r'\b\w+\b', question.lower()))
        sentences = re.split(r'[.!?]+', combined_content)
        
        # Score sentences based on question word overlap
        scored_sentences = []
        for sentence in sentences:
            if len(sentence.strip()) < 20:  # Skip very short sentences
                continue
            
            sentence_words = set(re.findall(r'\b\w+\b', sentence.lower()))
            overlap = len(question_words.intersection(sentence_words))
            
            if overlap > 0:
                scored_sentences.append((overlap, sentence.strip()))
        
        # Sort by score and take top sentences
        scored_sentences.sort(key=lambda x: x[0], reverse=True)
        
        if scored_sentences:
            # Combine top 2-3 sentences for the answer
            answer_parts = [sent[1] for sent in scored_sentences[:3]]
            answer = ". ".join(answer_parts)
            
            # Clean up the answer
            if not answer.endswith('.'):
                answer += '.'
            
            return answer
        else:
            # Fallback: return first part of most relevant document
            return relevant_docs[0]['content'][:300] + "..."
    
    def run_accuracy_test(self) -> Dict:
        """Run accuracy test using actual RAGAgent methods"""
        print("\n📊 Running Accuracy Test")
        print("-" * 40)
        
        # Test questions based on your actual documents
        test_questions = [
            {
                'question': 'What is ESG?',
                'expected_keywords': ['environmental', 'social', 'governance', 'esg'],
                'category': 'basic'
            },
            {
                'question': 'What are ESG disclosures?',
                'expected_keywords': ['disclosure', 'reporting', 'transparency', 'stakeholders'],
                'category': 'basic'
            },
            {
                'question': 'How do companies report ESG information?',
                'expected_keywords': ['report', 'companies', 'information', 'framework'],
                'category': 'intermediate'
            },
            {
                'question': 'What are the benefits of ESG programs?',
                'expected_keywords': ['benefits', 'programs', 'value', 'performance'],
                'category': 'intermediate'
            },
            {
                'question': 'What frameworks are used for ESG reporting?',
                'expected_keywords': ['framework', 'reporting', 'standards', 'guidelines'],
                'category': 'advanced'
            }
        ]
        
        results = []
        total_score = 0
        
        for i, test in enumerate(test_questions, 1):
            print(f"\n   Question {i}: {test['question']}")
            
            try:
                # Use the actual search method from RAGAgent
                relevant_docs = self.rag_agent.search(test['question'], top_k=3)
                
                # Generate answer using local processing
                answer = self.local_answer_generation(test['question'], relevant_docs)
                
                # Calculate accuracy based on keyword presence
                found_keywords = 0
                answer_lower = answer.lower()
                
                for keyword in test['expected_keywords']:
                    if keyword.lower() in answer_lower:
                        found_keywords += 1
                
                accuracy = found_keywords / len(test['expected_keywords'])
                total_score += accuracy
                
                print(f"      Answer: {answer[:100]}...")
                print(f"      Keywords found: {found_keywords}/{len(test['expected_keywords'])}")
                print(f"      Accuracy: {accuracy:.2%}")
                
                results.append({
                    'question': test['question'],
                    'answer': answer,
                    'accuracy': accuracy,
                    'category': test['category'],
                    'keywords_found': found_keywords,
                    'total_keywords': len(test['expected_keywords']),
                    'sources': [doc['document_id'] for doc in relevant_docs]
                })
                
            except Exception as e:
                print(f"      ❌ Error: {e}")
                results.append({
                    'question': test['question'],
                    'answer': f'ERROR: {str(e)}',
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
    
    def apply_optimizations(self):
        """Apply simple optimizations"""
        print("\n🔧 Applying Optimizations")
        print("-" * 40)
        
        # Optimization 1: Enhance search with query expansion
        print("1. Enhancing search with query expansion...")
        original_search = self.rag_agent.search
        
        def enhanced_search(query: str, top_k: int = 5):
            # Add related terms for better search
            query_expansions = {
                'esg': 'environmental social governance sustainability',
                'environmental': 'climate carbon emissions energy',
                'social': 'diversity inclusion community employees',
                'governance': 'board oversight compliance ethics',
                'disclosure': 'reporting transparency communication',
                'framework': 'standards guidelines methodology'
            }
            
            # Expand query with related terms
            expanded_query = query
            for term, expansion in query_expansions.items():
                if term in query.lower():
                    expanded_query += f" {expansion}"
            
            return original_search(expanded_query, top_k)
        
        self.rag_agent.search = enhanced_search
        print("   ✅ Search enhancement applied")
        
        # Optimization 2: Improve answer generation
        print("2. Improving answer generation...")
        original_gen = self.local_answer_generation
        
        def enhanced_generation(question: str, relevant_docs: List[Dict]) -> str:
            if not relevant_docs:
                return "I don't have relevant information to answer your ESG-related question."
            
            # Enhanced processing based on question type
            question_lower = question.lower()
            
            if any(word in question_lower for word in ['what is', 'define']):
                return self._generate_definition_answer(question, relevant_docs)
            elif any(word in question_lower for word in ['how', 'process', 'steps']):
                return self._generate_process_answer(question, relevant_docs)
            elif any(word in question_lower for word in ['benefits', 'advantages']):
                return self._generate_benefits_answer(question, relevant_docs)
            else:
                return original_gen(question, relevant_docs)
        
        self.local_answer_generation = enhanced_generation
        print("   ✅ Answer generation enhancement applied")
        
        print("✅ Optimizations completed")
    
    def _generate_definition_answer(self, question: str, docs: List[Dict]) -> str:
        """Generate definition-focused answers"""
        combined_content = " ".join([doc['content'] for doc in docs])
        
        # Look for definition sentences
        sentences = re.split(r'[.!?]+', combined_content)
        question_words = set(re.findall(r'\b\w+\b', question.lower()))
        
        # Find sentences that likely contain definitions
        definition_sentences = []
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(pattern in sentence_lower for pattern in ['is', 'are', 'refers to', 'means', 'stands for']):
                sentence_words = set(re.findall(r'\b\w+\b', sentence_lower))
                if len(question_words.intersection(sentence_words)) >= 1:
                    definition_sentences.append(sentence.strip())
        
        if definition_sentences:
            return ". ".join(definition_sentences[:2]) + "."
        else:
            return docs[0]['content'][:200] + "..."
    
    def _generate_process_answer(self, question: str, docs: List[Dict]) -> str:
        """Generate process-focused answers"""
        combined_content = " ".join([doc['content'] for doc in docs])
        
        # Look for process-related sentences
        process_keywords = ['steps', 'process', 'how', 'method', 'approach', 'way']
        sentences = re.split(r'[.!?]+', combined_content)
        
        process_sentences = []
        for sentence in sentences:
            if any(keyword in sentence.lower() for keyword in process_keywords):
                if len(sentence.strip()) > 20:
                    process_sentences.append(sentence.strip())
        
        if process_sentences:
            return ". ".join(process_sentences[:3]) + "."
        else:
            return docs[0]['content'][:300] + "..."
    
    def _generate_benefits_answer(self, question: str, docs: List[Dict]) -> str:
        """Generate benefits-focused answers"""
        combined_content = " ".join([doc['content'] for doc in docs])
        
        # Look for benefit-related sentences
        benefit_keywords = ['benefit', 'advantage', 'value', 'help', 'improve', 'enhance']
        sentences = re.split(r'[.!?]+', combined_content)
        
        benefit_sentences = []
        for sentence in sentences:
            if any(keyword in sentence.lower() for keyword in benefit_keywords):
                if len(sentence.strip()) > 20:
                    benefit_sentences.append(sentence.strip())
        
        if benefit_sentences:
            return ". ".join(benefit_sentences[:3]) + "."
        else:
            return docs[0]['content'][:300] + "..."
    
    def boost_accuracy(self) -> Dict:
        """Main accuracy boosting process"""
        print("\n🎯 ACCURACY BOOSTING PROCESS")
        print("=" * 50)
        
        # Step 1: Initial test
        print("📊 Step 1: Initial Accuracy Test")
        initial_results = self.run_accuracy_test()
        initial_accuracy = initial_results['overall_accuracy']
        
        # Step 2: Apply optimizations if needed
        if initial_accuracy < self.target_accuracy:
            print(f"\n🔧 Step 2: Applying Optimizations (Current: {initial_accuracy:.2%} < Target: {self.target_accuracy:.2%})")
            self.apply_optimizations()
            
            # Step 3: Re-test
            print("\n📊 Step 3: Post-Optimization Test")
            final_results = self.run_accuracy_test()
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
        filename = f"working_accuracy_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n💾 Results saved to: {filename}")


def main():
    """Main function"""
    print("🎯 Working Accuracy Booster for RAG Bot")
    print("Using actual RAGAgent methods - No API dependencies")
    print("=" * 50)
    
    try:
        # Create booster
        booster = WorkingAccuracyBooster()
        
        # Initialize
        booster.initialize_system()
        
        # Boost accuracy
        results = booster.boost_accuracy()
        
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
            print(f"   📊 Current accuracy: {results['final_accuracy']:.2%} (Target: 90%)")
        
        # Save results
        booster.save_results(results)
        
        print(f"\n✅ Working accuracy boosting completed!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()