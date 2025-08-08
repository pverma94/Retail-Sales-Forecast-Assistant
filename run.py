#!/usr/bin/env python3
"""
Retail Sales Forecast Assistant - Optimized Main Startup Script
This script helps you run the optimized application with better performance and accuracy.
"""

import subprocess
import sys
import os
import time

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        'streamlit', 'flask', 'flask_cors', 'pandas', 'numpy',
        'tensorflow', 'plotly', 'scikit-learn', 'joblib'
    ]

    missing_packages = []

    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n📦 Installing missing packages...")
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing_packages)
        print("✅ All packages installed successfully!")
    else:
        print("✅ All required packages are installed!")

def check_dataset():
    """Check if the reduced dataset exists"""
    if not os.path.exists('data/superstore_reduced_dataset.csv'):
        print("📊 Creating reduced dataset...")
        try:
            subprocess.check_call([sys.executable, "utils/data_processor.py", "reduce", "data/superstore_final_dataset.csv", "data/superstore_reduced_dataset.csv"])
            print("✅ Reduced dataset created successfully!")
        except subprocess.CalledProcessError:
            print("❌ Error creating reduced dataset. Please check if data/superstore_final_dataset.csv exists.")
            return False
    else:
        print("✅ Reduced dataset already exists!")
    return True

def run_frontend_only():
    """Run only the optimized Streamlit frontend"""
    print("\n🚀 Starting Optimized Retail Sales Forecast Assistant (Frontend Only)...")
    print("📱 Opening web interface...")
    print("🌐 The application will open in your default browser.")
    print("📋 If it doesn't open automatically, go to: http://localhost:8501")
    print("\n🎯 **Optimized Features:**")
    print("   - Enhanced LSTM architecture with batch normalization")
    print("   - Robust scaling for better outlier handling")
    print("   - Feature engineering with seasonal patterns")
    print("   - Model performance evaluation")
    print("   - Memory optimization and caching")
    print("\n⏹️  Press Ctrl+C to stop the application")

    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "frontend/app.py"])
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except Exception as e:
        print(f"❌ Error running application: {e}")

def run_backend_only():
    """Run only the optimized Flask backend"""
    print("\n🔧 Starting Optimized Flask Backend API...")
    print("🌐 API will be available at: http://localhost:5000")
    print("\n🎯 **API Endpoints:**")
    print("   - GET  /health - Health check")
    print("   - POST /forecast - Generate optimized forecast")
    print("   - POST /analyze - Analyze data")
    print("\n⏹️  Press Ctrl+C to stop the backend")

    try:
        subprocess.run([sys.executable, "backend/api.py"])
    except KeyboardInterrupt:
        print("\n👋 Backend stopped by user")
    except Exception as e:
        print(f"❌ Error running backend: {e}")

def run_full_stack():
    """Run both optimized frontend and backend"""
    print("\n🚀 Starting Full Stack Optimized Application...")
    print("🔧 Backend API: http://localhost:5000")
    print("📱 Frontend: http://localhost:8501")
    print("\n🎯 **Optimized Features:**")
    print("   - Enhanced LSTM with 3-layer architecture")
    print("   - Robust scaling and feature engineering")
    print("   - Model validation and performance metrics")
    print("   - Memory optimization and efficient processing")
    print("\n⏹️  Press Ctrl+C to stop both services")

    try:
        # Start backend in background
        backend_process = subprocess.Popen([sys.executable, "backend/api.py"])

        # Wait a moment for backend to start
        time.sleep(3)

        # Start frontend
        frontend_process = subprocess.Popen([sys.executable, "-m", "streamlit", "run", "frontend/app.py"])

        # Wait for either process to finish
        backend_process.wait()
        frontend_process.wait()

    except KeyboardInterrupt:
        print("\n👋 Stopping all services...")
        if 'backend_process' in locals():
            backend_process.terminate()
        if 'frontend_process' in locals():
            frontend_process.terminate()
    except Exception as e:
        print(f"❌ Error running full stack: {e}")

def show_optimization_info():
    """Show information about the optimizations"""
    print("\n🔍 **Optimization Details:**")
    print("=" * 50)
    print("📊 **Performance Improvements:**")
    print("   • Enhanced LSTM Architecture: 3-layer with batch normalization")
    print("   • Robust Scaling: Better outlier handling than MinMax scaling")
    print("   • Feature Engineering: Seasonal features, rolling statistics")
    print("   • Memory Optimization: Efficient data processing and caching")
    print("   • Model Validation: Cross-validation and performance metrics")
    
    print("\n🎯 **Accuracy Improvements:**")
    print("   • Better hyperparameter tuning with early stopping")
    print("   • Learning rate scheduling for optimal convergence")
    print("   • Enhanced feature set with temporal patterns")
    print("   • Model confidence assessment")
    print("   • Advanced alert system with trend detection")
    
    print("\n⚡ **Code Optimizations:**")
    print("   • Removed unnecessary files and dependencies")
    print("   • Consolidated duplicate code")
    print("   • Added type hints for better code quality")
    print("   • Improved error handling and logging")
    print("   • Enhanced API response structure")

def main():
    """Main function"""
    print("=" * 70)
    print("📊 Optimized Retail Sales Forecast Assistant")
    print("=" * 70)

    # Check dependencies
    print("\n🔍 Checking dependencies...")
    check_dependencies()

    # Check dataset
    print("\n📊 Checking dataset...")
    if not check_dataset():
        return

    # Show optimization info
    show_optimization_info()

    # Ask user for run mode
    print("\n🎯 Choose run mode:")
    print("1. Frontend Only (Recommended for demo)")
    print("2. Backend Only (API server)")
    print("3. Full Stack (Frontend + Backend)")
    print("4. Exit")

    while True:
        try:
            choice = input("\nEnter your choice (1-4): ").strip()

            if choice == "1":
                run_frontend_only()
                break
            elif choice == "2":
                run_backend_only()
                break
            elif choice == "3":
                run_full_stack()
                break
            elif choice == "4":
                print("👋 Goodbye!")
                break
            else:
                print("❌ Invalid choice. Please enter 1, 2, 3, or 4.")
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break

if __name__ == "__main__":
    main()
