import argparse
import sys
import os

def run_cli():
    """Run the command-line interface version"""
    from improved_web_search import setup_and_run
    
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        print("Enter your question: ", end="")
        question = input()
    
    if not question:
        print("Please provide a question.")
        return
    
    answer = setup_and_run(question)
    print("\n=== FINAL ANSWER ===")
    print(answer)

def run_streamlit():
    """Run the Streamlit web interface"""
    try:
        import streamlit
        os.system("streamlit run streamlit_app.py")
    except ImportError:
        print("Streamlit is not installed. Install it with 'pip install streamlit'.")
        sys.exit(1)

def main():
    """Main function to run the application in different modes"""
    parser = argparse.ArgumentParser(description="Corrective RAG with Web Search")
    parser.add_argument("--web", action="store_true", help="Run the web interface using Streamlit")
    parser.add_argument("--cli", action="store_true", help="Run the command-line interface")
    parser.add_argument("question", nargs="*", help="Question to ask (CLI mode only)")
    
    args = parser.parse_args()
    
    # Check for API keys
    if "GOOGLE_API_KEY" not in os.environ or os.environ["GOOGLE_API_KEY"] == "YOUR_GOOGLE_API_KEY":
        print("Warning: GOOGLE_API_KEY not set. Set it in your environment variables.")
    
    if "TAVILY_API_KEY" not in os.environ or os.environ["TAVILY_API_KEY"] == "YOUR_TAVILY_API_KEY":
        print("Warning: TAVILY_API_KEY not set. Set it in your environment variables.")
    
    # Determine which mode to run in
    if args.web:
        run_streamlit()
    elif args.cli or args.question:
        run_cli()
    else:
        # Default to Streamlit if no arguments are given
        print("Starting web interface. Use --cli for command-line mode.")
        run_streamlit()

if __name__ == "__main__":
    main()
