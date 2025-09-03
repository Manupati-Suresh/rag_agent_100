#!/usr/bin/env python3
"""
Simple Script to Achieve 90% Accuracy
====================================
This script runs the complete accuracy improvement pipeline to reach 90% accuracy.
Uses the working systems we've built and tested.
"""

import os
import sys
import json
import time
from datetime import datetime

def main():
    """Run the complete 90% accuracy improvement pipeline"""
    
    print("🚀 Starting 90% Accuracy Achievement Pipeline")
    print("=" * 50)
    
    # Step 1: Run baseline accuracy test
    print("\n📊 Step 1: Running baseline accuracy test...")
    try:
        os.system("python working_accuracy_booster.py")
        print("✅ Baseline test completed")
    except Exception as e:
        print(f"❌ Baseline test failed: {e}")
        return False
    
    # Step 2: Run advanced prompt engineering
    print("\n🧠 Step 2: Running advanced prompt engineering...")
    try:
        os.system("python advanced_prompt_engineering.py")
        print("✅ Advanced prompting completed")
    except Exception as e:
        print(f"❌ Advanced prompting failed: {e}")
        return False
    
    # Step 3: Run accuracy optimizer
    print("\n⚡ Step 3: Running accuracy optimizer...")
    try:
        os.system("python accuracy_optimizer.py")
        print("✅ Accuracy optimization completed")
    except Exception as e:
        print(f"❌ Accuracy optimization failed: {e}")
        return False
    
    # Step 4: Run final accuracy test
    print("\n🎯 Step 4: Running final accuracy test...")
    try:
        os.system("python accuracy_test_suite.py")
        print("✅ Final accuracy test completed")
    except Exception as e:
        print(f"❌ Final test failed: {e}")
        return False
    
    # Step 5: Check if we achieved 90%
    print("\n📈 Step 5: Checking final accuracy...")
    try:
        # Look for the latest results file
        result_files = [f for f in os.listdir('.') if f.startswith('accuracy_test_results_') and f.endswith('.json')]
        if result_files:
            latest_file = sorted(result_files)[-1]
            with open(latest_file, 'r') as f:
                results = json.load(f)
            
            overall_accuracy = results.get('overall_accuracy', 0)
            print(f"🎯 Final Accuracy: {overall_accuracy:.1f}%")
            
            if overall_accuracy >= 90:
                print("🎉 SUCCESS! 90% accuracy achieved!")
                return True
            else:
                print(f"⚠️  Close! Need {90 - overall_accuracy:.1f}% more to reach 90%")
                print("💡 Consider adding more ESG documents or running additional optimization")
                return False
        else:
            print("❌ No results file found")
            return False
            
    except Exception as e:
        print(f"❌ Error checking results: {e}")
        return False

if __name__ == "__main__":
    print("🎯 90% Accuracy Achievement Script")
    print("This script will run all optimization steps to achieve 90% accuracy")
    print()
    
    # Ask for confirmation
    response = input("Ready to start? (y/n): ").lower().strip()
    if response != 'y':
        print("Cancelled.")
        sys.exit(0)
    
    start_time = time.time()
    success = main()
    end_time = time.time()
    
    print("\n" + "=" * 50)
    print(f"⏱️  Total time: {end_time - start_time:.1f} seconds")
    
    if success:
        print("🎉 MISSION ACCOMPLISHED! 90% accuracy achieved!")
    else:
        print("📈 Progress made! Run again or add more documents for 90%")
    
    print("=" * 50)