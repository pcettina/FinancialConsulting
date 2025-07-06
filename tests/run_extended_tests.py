# Run Extended GPFA Tests
"""
Simple extended testing script for the GPFA prediction model.
Runs multiple test scenarios to validate system performance.
"""

import time
import json
from datetime import datetime
from real_data_integration import EnhancedGPFAPredictor

def run_extended_test_scenarios():
    """Run multiple extended test scenarios"""
    
    print("🚀 Starting Extended GPFA Testing...")
    print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Test symbols
    symbols = ['AAPL', 'GOOGL', 'MSFT']
    
    # Test scenarios configuration
    test_scenarios = [
        {
            'name': 'Quick Validation Test',
            'duration_minutes': 2,
            'description': 'Short validation test'
        },
        {
            'name': 'Extended Duration Test', 
            'duration_minutes': 8,
            'description': 'Longer duration test'
        },
        {
            'name': 'High Frequency Test',
            'duration_minutes': 5,
            'description': 'High frequency testing'
        },
        {
            'name': 'Market Hours Test',
            'duration_minutes': 12,
            'description': 'Market hours monitoring'
        }
    ]
    
    # Results storage
    all_results = {
        'test_timestamp': datetime.now().isoformat(),
        'symbols_tested': symbols,
        'scenarios': {},
        'overall_performance': {}
    }
    
    successful_scenarios = 0
    total_cycles = 0
    total_predictions = 0
    
    # Run each test scenario
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n🔍 Running Scenario {i}: {scenario['name']}")
        print(f"   Duration: {scenario['duration_minutes']} minutes")
        print(f"   Description: {scenario['description']}")
        
        try:
            # Initialize GPFA predictor
            gpfa_predictor = EnhancedGPFAPredictor(symbols, n_factors=3)
            
            # Run the test
            start_time = time.time()
            results = gpfa_predictor.run_real_time_prediction(
                duration_minutes=scenario['duration_minutes']
            )
            end_time = time.time()
            
            # Add scenario metadata
            results['scenario_name'] = scenario['name']
            results['scenario_duration'] = scenario['duration_minutes']
            results['test_duration_actual'] = end_time - start_time
            results['scenario_number'] = i
            
            # Store results
            all_results['scenarios'][scenario['name'].lower().replace(' ', '_')] = results
            
            if results.get('success', False):
                successful_scenarios += 1
                total_cycles += results.get('cycles_completed', 0)
                total_predictions += results.get('predictions_made', 0)
                
                print(f"   ✅ SUCCESS: {results['cycles_completed']} cycles, {results['predictions_made']} predictions")
            else:
                print(f"   ❌ FAILED: {results.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"   ❌ ERROR: {str(e)}")
            all_results['scenarios'][scenario['name'].lower().replace(' ', '_')] = {
                'success': False,
                'error': str(e),
                'scenario_name': scenario['name']
            }
    
    # Calculate overall performance
    total_scenarios = len(test_scenarios)
    success_rate = (successful_scenarios / total_scenarios) * 100 if total_scenarios > 0 else 0
    
    all_results['overall_performance'] = {
        'total_scenarios': total_scenarios,
        'successful_scenarios': successful_scenarios,
        'success_rate': round(success_rate, 2),
        'total_cycles': total_cycles,
        'total_predictions': total_predictions,
        'average_cycles_per_scenario': round(total_cycles / total_scenarios, 1) if total_scenarios > 0 else 0,
        'average_predictions_per_scenario': round(total_predictions / total_scenarios, 1) if total_scenarios > 0 else 0
    }
    
    return all_results

def display_extended_results(results):
    """Display comprehensive extended test results"""
    
    print("\n" + "="*70)
    print("🎯 EXTENDED GPFA TESTING RESULTS")
    print("="*70)
    
    # Test overview
    print(f"📅 Test Date: {results.get('test_timestamp', 'Unknown')[:10]}")
    print(f"📊 Symbols Tested: {', '.join(results.get('symbols_tested', []))}")
    
    # Overall performance
    overall = results.get('overall_performance', {})
    if overall:
        print(f"\n📈 OVERALL PERFORMANCE")
        print("-" * 40)
        print(f"✅ Successful Scenarios: {overall.get('successful_scenarios', 0)}")
        print(f"📊 Success Rate: {overall.get('success_rate', 0)}%")
        print(f"🔄 Total Cycles: {overall.get('total_cycles', 0)}")
        print(f"🎯 Total Predictions: {overall.get('total_predictions', 0)}")
        print(f"📊 Avg Cycles/Scenario: {overall.get('average_cycles_per_scenario', 0)}")
        print(f"📊 Avg Predictions/Scenario: {overall.get('average_predictions_per_scenario', 0)}")
    
    # Individual scenario results
    scenarios = results.get('scenarios', {})
    if scenarios:
        print(f"\n🔍 SCENARIO DETAILS")
        print("-" * 40)
        
        for scenario_key, scenario_results in scenarios.items():
            scenario_name = scenario_results.get('scenario_name', scenario_key)
            print(f"\n📋 {scenario_name}:")
            
            if scenario_results.get('success', False):
                print(f"  ✅ Status: SUCCESS")
                print(f"  ⏱️  Duration: {scenario_results.get('scenario_duration', 0)} minutes")
                print(f"  🔄 Cycles: {scenario_results.get('cycles_completed', 0)}")
                print(f"  🎯 Predictions: {scenario_results.get('predictions_made', 0)}")
                print(f"  ⏱️  Actual Time: {scenario_results.get('test_duration_actual', 0):.1f} seconds")
            else:
                print(f"  ❌ Status: FAILED")
                print(f"  🚨 Error: {scenario_results.get('error', 'Unknown error')}")
    
    # Performance assessment
    print(f"\n🎯 PERFORMANCE ASSESSMENT")
    print("-" * 40)
    
    success_rate = overall.get('success_rate', 0)
    if success_rate >= 90:
        print("🟢 EXCELLENT: System is ready for production deployment")
        print("   • All scenarios passed successfully")
        print("   • System stability is high")
        print("   • Ready for live trading implementation")
    elif success_rate >= 75:
        print("🟡 GOOD: System is mostly ready with minor improvements needed")
        print("   • Most scenarios passed successfully")
        print("   • Consider addressing failed scenarios")
        print("   • Ready for extended testing")
    elif success_rate >= 50:
        print("🟠 FAIR: System needs improvements before production")
        print("   • Several scenarios need attention")
        print("   • Review error logs and system configuration")
        print("   • Consider additional testing")
    else:
        print("🔴 POOR: System requires significant improvements")
        print("   • Multiple scenarios failed")
        print("   • Review system architecture and implementation")
        print("   • Extensive testing and debugging required")
    
    # Next steps
    print(f"\n🚀 NEXT STEPS")
    print("-" * 40)
    print("1. ✅ Extended testing completed")
    print("2. 🔄 Review individual scenario results")
    print("3. 🔄 Address any failed scenarios")
    print("4. 🔄 Consider production deployment if success rate > 90%")
    print("5. 🔄 Monitor system performance in production")

def main():
    """Main function to run extended tests"""
    
    # Run extended test scenarios
    results = run_extended_test_scenarios()
    
    # Display results
    display_extended_results(results)
    
    # Save results
    if results:
        with open('extended_gpfa_test_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n💾 Extended test results saved to: extended_gpfa_test_results.json")
    
    print(f"\nTest completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main() 