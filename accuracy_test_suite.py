#!/usr/bin/env python3
"""
Comprehensive Accuracy Test Suite for RAG Bot
Advanced testing framework to measure and improve accuracy to 90%+
"""

import asyncio
import json
import time
from datetime import datetime
from typing import List, Dict, Tuple
import numpy as np
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
import seaborn as sns
from rag_agent_telegram import create_telegram_rag_agent
from accuracy_optimizer import AccuracyOptimizer, run_accuracy_optimization

@dataclass
class TestResult:
    """Individual test result data structure"""
    question_id: int
    question: str
    category: str
    difficulty: str
    expected_answer: str
    generated_answer: str
    accuracy_score: float
    response_time: float
    sources_count: int
    confidence_score: float
    error_type: str
    timestamp: str

class AccuracyTestSuite:
    """Comprehensive accuracy testing framework"""
    
    def __init__(self, rag_agent):
        self.rag_agent = rag_agent
        self.test_results = []
        self.accuracy_threshold = 0.90
        
        # Test categories with different difficulty levels - ESG FOCUSED
        self.test_categories = {
            'esg_basic': {
                'description': 'Basic ESG questions',
                'weight': 0.2,
                'questions': [
                    {
                        'question': 'What is ESG?',
                        'expected_keywords': ['environmental', 'social', 'governance', 'ESG'],
                        'expected_concepts': ['sustainability', 'responsibility', 'framework']
                    },
                    {
                        'question': 'What are ESG disclosures?',
                        'expected_keywords': ['disclosures', 'reporting', 'transparency', 'ESG'],
                        'expected_concepts': ['stakeholders', 'information', 'performance']
                    },
                    {
                        'question': 'Why are ESG programs important?',
                        'expected_keywords': ['programs', 'important', 'benefits', 'ESG'],
                        'expected_concepts': ['risk management', 'reputation', 'compliance']
                    }
                ]
            },
            'esg_intermediate': {
                'description': 'Intermediate ESG questions',
                'weight': 0.25,
                'questions': [
                    {
                        'question': 'How do companies implement ESG programs?',
                        'expected_keywords': ['implement', 'companies', 'programs', 'ESG'],
                        'expected_concepts': ['strategy', 'framework', 'management']
                    },
                    {
                        'question': 'What are the key components of ESG reporting?',
                        'expected_keywords': ['components', 'reporting', 'ESG', 'key'],
                        'expected_concepts': ['metrics', 'standards', 'disclosure']
                    },
                    {
                        'question': 'How do ESG factors affect business performance?',
                        'expected_keywords': ['factors', 'affect', 'business', 'performance'],
                        'expected_concepts': ['impact', 'value', 'sustainability']
                    }
                ]
            },
            'esg_analytical': {
                'description': 'ESG analytical and comparison questions',
                'weight': 0.25,
                'questions': [
                    {
                        'question': 'Compare environmental and social aspects of ESG',
                        'expected_keywords': ['environmental', 'social', 'compare', 'aspects'],
                        'expected_concepts': ['differences', 'similarities', 'impact']
                    },
                    {
                        'question': 'What are the advantages and challenges of ESG implementation?',
                        'expected_keywords': ['advantages', 'challenges', 'implementation', 'ESG'],
                        'expected_concepts': ['benefits', 'obstacles', 'solutions']
                    },
                    {
                        'question': 'How does ESG reporting benefit different stakeholders?',
                        'expected_keywords': ['reporting', 'benefit', 'stakeholders', 'ESG'],
                        'expected_concepts': ['investors', 'customers', 'employees']
                    }
                ]
            },
            'esg_procedural': {
                'description': 'ESG procedural and how-to questions',
                'weight': 0.15,
                'questions': [
                    {
                        'question': 'How to start an ESG program in a company?',
                        'expected_keywords': ['start', 'program', 'company', 'ESG'],
                        'expected_concepts': ['steps', 'planning', 'implementation']
                    },
                    {
                        'question': 'Steps to create effective ESG disclosures',
                        'expected_keywords': ['steps', 'create', 'effective', 'disclosures'],
                        'expected_concepts': ['process', 'guidelines', 'best practices']
                    }
                ]
            },
            'esg_advanced': {
                'description': 'Advanced ESG conceptual questions',
                'weight': 0.15,
                'questions': [
                    {
                        'question': 'Explain the concept of ESG materiality',
                        'expected_keywords': ['concept', 'materiality', 'ESG', 'explain'],
                        'expected_concepts': ['significance', 'relevance', 'impact']
                    },
                    {
                        'question': 'What is the role of governance in ESG frameworks?',
                        'expected_keywords': ['role', 'governance', 'frameworks', 'ESG'],
                        'expected_concepts': ['oversight', 'accountability', 'structure']
                    }
                ]
            }
        }
    
    async def run_comprehensive_accuracy_test(self) -> Dict:
        """Run comprehensive accuracy testing across all categories"""
        print("🧪 Starting Comprehensive Accuracy Test Suite")
        print("=" * 60)
        
        start_time = time.time()
        all_results = []
        category_scores = {}
        
        # Test each category
        for category_name, category_data in self.test_categories.items():
            print(f"\n📋 Testing Category: {category_data['description']}")
            print(f"   Weight: {category_data['weight']:.0%}")
            
            category_results = []
            
            for i, test_case in enumerate(category_data['questions'], 1):
                print(f"   Question {i}/{len(category_data['questions'])}: {test_case['question'][:50]}...")
                
                # Run the test
                result = await self._run_single_test(
                    question_id=len(all_results) + 1,
                    question=test_case['question'],
                    category=category_name,
                    expected_keywords=test_case['expected_keywords'],
                    expected_concepts=test_case.get('expected_concepts', [])
                )
                
                category_results.append(result)
                all_results.append(result)
                
                print(f"      Accuracy: {result.accuracy_score:.2%} | Time: {result.response_time:.2f}s")
            
            # Calculate category average
            category_avg = np.mean([r.accuracy_score for r in category_results])
            category_scores[category_name] = {
                'average_accuracy': category_avg,
                'weight': category_data['weight'],
                'weighted_score': category_avg * category_data['weight']
            }
            
            print(f"   Category Average: {category_avg:.2%}")
        
        # Calculate overall weighted accuracy
        overall_accuracy = sum(scores['weighted_score'] for scores in category_scores.values())
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Compile comprehensive results
        test_summary = {
            'overall_accuracy': overall_accuracy,
            'target_accuracy': self.accuracy_threshold,
            'accuracy_achieved': overall_accuracy >= self.accuracy_threshold,
            'total_questions': len(all_results),
            'total_time': total_time,
            'average_response_time': np.mean([r.response_time for r in all_results]),
            'category_breakdown': category_scores,
            'detailed_results': [asdict(result) for result in all_results],
            'timestamp': datetime.now().isoformat()
        }
        
        # Save results
        await self._save_test_results(test_summary)
        
        # Display summary
        self._display_test_summary(test_summary)
        
        return test_summary
    
    async def _run_single_test(self, question_id: int, question: str, category: str, 
                              expected_keywords: List[str], expected_concepts: List[str]) -> TestResult:
        """Run a single accuracy test"""
        start_time = time.time()
        
        try:
            # Get response from RAG agent
            response = self.rag_agent.chat_with_documents(question)
            
            end_time = time.time()
            response_time = end_time - start_time
            
            if response['success']:
                generated_answer = response['response']
                sources = response.get('sources', [])
                
                # Calculate accuracy score
                accuracy_score = self._calculate_comprehensive_accuracy(
                    question, generated_answer, expected_keywords, expected_concepts
                )
                
                # Calculate confidence score
                confidence_score = self._calculate_confidence_score(generated_answer, sources)
                
                return TestResult(
                    question_id=question_id,
                    question=question,
                    category=category,
                    difficulty='intermediate',
                    expected_answer=str(expected_keywords + expected_concepts),
                    generated_answer=generated_answer,
                    accuracy_score=accuracy_score,
                    response_time=response_time,
                    sources_count=len(sources),
                    confidence_score=confidence_score,
                    error_type='none',
                    timestamp=datetime.now().isoformat()
                )
            else:
                # Failed response
                return TestResult(
                    question_id=question_id,
                    question=question,
                    category=category,
                    difficulty='intermediate',
                    expected_answer=str(expected_keywords + expected_concepts),
                    generated_answer='ERROR: ' + response.get('error', 'Unknown error'),
                    accuracy_score=0.0,
                    response_time=response_time,
                    sources_count=0,
                    confidence_score=0.0,
                    error_type='generation_failed',
                    timestamp=datetime.now().isoformat()
                )
        
        except Exception as e:
            end_time = time.time()
            return TestResult(
                question_id=question_id,
                question=question,
                category=category,
                difficulty='intermediate',
                expected_answer=str(expected_keywords + expected_concepts),
                generated_answer=f'EXCEPTION: {str(e)}',
                accuracy_score=0.0,
                response_time=end_time - start_time,
                sources_count=0,
                confidence_score=0.0,
                error_type='exception',
                timestamp=datetime.now().isoformat()
            )
    
    def _calculate_comprehensive_accuracy(self, question: str, answer: str, 
                                        expected_keywords: List[str], 
                                        expected_concepts: List[str]) -> float:
        """Calculate comprehensive accuracy score"""
        if not answer or answer.startswith('ERROR') or answer.startswith('EXCEPTION'):
            return 0.0
        
        answer_lower = answer.lower()
        
        # 1. Keyword Coverage (40%)
        keyword_score = 0.0
        if expected_keywords:
            found_keywords = sum(1 for kw in expected_keywords if kw.lower() in answer_lower)
            keyword_score = found_keywords / len(expected_keywords)
        
        # 2. Concept Coverage (30%)
        concept_score = 0.0
        if expected_concepts:
            found_concepts = sum(1 for concept in expected_concepts if concept.lower() in answer_lower)
            concept_score = found_concepts / len(expected_concepts)
        
        # 3. Answer Completeness (20%)
        completeness_score = min(1.0, len(answer.split()) / 50)  # Normalize to 50 words
        
        # 4. Relevance to Question (10%)
        question_words = set(question.lower().split())
        answer_words = set(answer_lower.split())
        relevance_score = len(question_words.intersection(answer_words)) / len(question_words)
        
        # Weighted combination
        total_score = (
            0.4 * keyword_score +
            0.3 * concept_score +
            0.2 * completeness_score +
            0.1 * relevance_score
        )
        
        return min(1.0, total_score)
    
    def _calculate_confidence_score(self, answer: str, sources: List[Dict]) -> float:
        """Calculate confidence score based on answer quality and sources"""
        if not answer or answer.startswith('ERROR'):
            return 0.0
        
        # Base confidence from answer length and structure
        base_confidence = min(1.0, len(answer.split()) / 30)
        
        # Source quality contribution
        source_confidence = 0.0
        if sources:
            avg_source_score = sum(s.get('score', 0) for s in sources) / len(sources)
            source_confidence = min(1.0, avg_source_score)
        
        # Combine scores
        return (base_confidence + source_confidence) / 2
    
    def _display_test_summary(self, summary: Dict):
        """Display comprehensive test summary"""
        print("\n" + "=" * 60)
        print("🎯 ACCURACY TEST RESULTS SUMMARY")
        print("=" * 60)
        
        print(f"📊 Overall Accuracy: {summary['overall_accuracy']:.2%}")
        print(f"🎯 Target Accuracy: {summary['target_accuracy']:.2%}")
        print(f"✅ Target Achieved: {'YES' if summary['accuracy_achieved'] else 'NO'}")
        print(f"📝 Total Questions: {summary['total_questions']}")
        print(f"⏱️ Total Time: {summary['total_time']:.1f}s")
        print(f"⚡ Avg Response Time: {summary['average_response_time']:.2f}s")
        
        print(f"\n📋 Category Breakdown:")
        for category, scores in summary['category_breakdown'].items():
            print(f"   {category}: {scores['average_accuracy']:.2%} (weight: {scores['weight']:.0%})")
        
        # Recommendations
        print(f"\n💡 Recommendations:")
        if summary['overall_accuracy'] < 0.90:
            print("   🔧 Run accuracy optimization: python -c 'from accuracy_optimizer import run_accuracy_optimization; import asyncio; asyncio.run(run_accuracy_optimization(your_agent))'")
            
            # Identify weak categories
            weak_categories = [cat for cat, scores in summary['category_breakdown'].items() 
                             if scores['average_accuracy'] < 0.80]
            if weak_categories:
                print(f"   📈 Focus on improving: {', '.join(weak_categories)}")
        else:
            print("   🎉 Excellent accuracy! Consider continuous monitoring.")
    
    async def _save_test_results(self, summary: Dict):
        """Save test results to file"""
        filename = f"accuracy_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Make summary JSON serializable
        serializable_summary = self._make_json_serializable(summary)
        
        with open(filename, 'w') as f:
            json.dump(serializable_summary, f, indent=2)
        
        print(f"\n💾 Results saved to: {filename}")
    
    def _make_json_serializable(self, obj):
        """Convert objects to JSON-serializable format"""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, bool):
            return obj
        elif isinstance(obj, (int, float, str)):
            return obj
        elif obj is None:
            return obj
        elif hasattr(obj, '__dict__'):
            return self._make_json_serializable(obj.__dict__)
        else:
            return str(obj)
    
    async def run_accuracy_improvement_cycle(self) -> Dict:
        """Run complete accuracy improvement cycle"""
        print("🔄 Starting Accuracy Improvement Cycle")
        print("=" * 50)
        
        # Step 1: Initial accuracy test
        print("1️⃣ Running initial accuracy assessment...")
        initial_results = await self.run_comprehensive_accuracy_test()
        initial_accuracy = initial_results['overall_accuracy']
        
        # Step 2: Run optimization if needed
        if initial_accuracy < self.accuracy_threshold:
            print(f"\n2️⃣ Accuracy below target ({initial_accuracy:.2%} < {self.accuracy_threshold:.2%})")
            print("Running accuracy optimization...")
            
            optimizer = AccuracyOptimizer(self.rag_agent, self.accuracy_threshold)
            optimization_results = await optimizer.optimize_for_accuracy()
            
            # Step 3: Re-test after optimization
            print("\n3️⃣ Re-testing after optimization...")
            final_results = await self.run_comprehensive_accuracy_test()
            final_accuracy = final_results['overall_accuracy']
            
            improvement_cycle_results = {
                'initial_accuracy': initial_accuracy,
                'final_accuracy': final_accuracy,
                'improvement': final_accuracy - initial_accuracy,
                'target_achieved': final_accuracy >= self.accuracy_threshold,
                'optimization_applied': True,
                'optimization_details': optimization_results,
                'initial_test_results': initial_results,
                'final_test_results': final_results
            }
        else:
            print(f"\n✅ Accuracy already meets target ({initial_accuracy:.2%} >= {self.accuracy_threshold:.2%})")
            improvement_cycle_results = {
                'initial_accuracy': initial_accuracy,
                'final_accuracy': initial_accuracy,
                'improvement': 0.0,
                'target_achieved': True,
                'optimization_applied': False,
                'initial_test_results': initial_results
            }
        
        # Save cycle results
        cycle_filename = f"accuracy_improvement_cycle_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        serializable_cycle_results = self._make_json_serializable(improvement_cycle_results)
        with open(cycle_filename, 'w') as f:
            json.dump(serializable_cycle_results, f, indent=2)
        
        print(f"\n💾 Improvement cycle results saved to: {cycle_filename}")
        
        return improvement_cycle_results


async def main():
    """Main function to run accuracy testing"""
    print("🎯 RAG Bot Accuracy Test Suite")
    print("Comprehensive testing to achieve 90%+ accuracy")
    print("=" * 50)
    
    # Initialize RAG agent
    print("🔄 Initializing RAG Agent...")
    agent = await create_telegram_rag_agent(use_telegram=True)
    
    # Create test suite
    test_suite = AccuracyTestSuite(agent)
    
    # Run improvement cycle
    results = await test_suite.run_accuracy_improvement_cycle()
    
    # Display final summary
    print("\n🎉 ACCURACY IMPROVEMENT CYCLE COMPLETED")
    print("=" * 50)
    print(f"Initial Accuracy: {results['initial_accuracy']:.2%}")
    print(f"Final Accuracy: {results['final_accuracy']:.2%}")
    print(f"Improvement: {results['improvement']:.2%}")
    print(f"Target Achieved: {'✅ YES' if results['target_achieved'] else '❌ NO'}")
    
    await agent.close()

if __name__ == "__main__":
    asyncio.run(main())