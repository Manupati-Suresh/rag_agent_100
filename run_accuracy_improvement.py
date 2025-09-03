#!/usr/bin/env python3
"""
Simple script to run RAG bot accuracy improvement to 90%
"""

import asyncio
from achieve_90_percent_accuracy import AccuracyAchievementSystem

async def quick_accuracy_boost():
    """Quick function to boost RAG accuracy to 90%"""
    print("🚀 Quick RAG Accuracy Boost to 90%")
    print("=" * 40)
    
    # Initialize system
    system = AccuracyAchievementSystem(target_accuracy=0.90)
    await system.initialize_system()
    
    # Run accuracy improvement
    results = await system.achieve_90_percent_accuracy()
    
    # Close system
    await system.rag_agent.close()
    
    return results

if __name__ == "__main__":
    print("🎯 Starting RAG Bot Accuracy Improvement...")
    
    try:
        results = asyncio.run(quick_accuracy_boost())
        
        if results['target_achieved']:
            print(f"\n🎉 SUCCESS! Achieved {results['final_accuracy']:.2%} accuracy!")
        else:
            print(f"\n⚠️ Reached {results['final_accuracy']:.2%} accuracy (target: 90%)")
            print("Consider adding more documents or running additional optimization")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure your Telegram credentials are configured in .env file")