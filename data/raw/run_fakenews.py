
import sys
from pathlib import Path
import logging
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from data.raw.generate_sample_data import generate_and_save_sample_data
from data.raw.analyze_fake_news import analyze_fake_news_dataset
from data.raw_data_pipeline import RawDataPipeline

logger = logging.getLogger(__name__)

def run_complete_fake_news_pipeline(generate_data: bool = False):
    """
    Run complete pipeline for fake_news.csv dataset
    
    Args:
        generate_data: Whether to generate sample data first
    """
    print("\n" + "="*60)
    print("FAKE NEWS DETECTION - DATA PIPELINE")
    print("="*60)
    
    start_time = datetime.now()
    
    # Step 1: Generate sample data if needed
    if generate_data:
        print("\n📁 Step 1: Generating sample dataset...")
        try:
            df = generate_and_save_sample_data()
            print(f"   ✓ Generated {len(df):,} samples")
            print(f"   ✓ Saved to data/raw/fake_news.csv")
        except Exception as e:
            print(f"   ✗ Error generating data: {e}")
            return
    
    # Step 2: Analyze dataset
    print("\n📊 Step 2: Analyzing dataset...")
    try:
        analysis_results = analyze_fake_news_dataset()
        print(f"   ✓ Analysis complete")
    except Exception as e:
        print(f"   ✗ Error analyzing dataset: {e}")
        return
    
    # Step 3: Run data processing pipeline
    print("\n⚙️  Step 3: Processing data pipeline...")
    try:
        pipeline = RawDataPipeline()
        
        # Since we only have fake_news.csv, use it as the primary dataset
        results = pipeline.run_complete_pipeline(
            datasets_to_load=["fake_news"],
            unification_strategy="concatenate",
            create_splits=True,
            save_intermediate=True
        )
        
        print(f"   ✓ Pipeline completed successfully")
        print(f"   ✓ Duration: {results.get('duration_seconds', 0):.1f} seconds")
        
        if "splits" in results:
            splits = results["splits"]
            print(f"\n   Dataset Splits:")
            for split_name, split_size in splits.items():
                print(f"   - {split_name}: {split_size:,} samples")
        
    except Exception as e:
        print(f"   ✗ Error in pipeline: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 4: Summary
    print("\n✅ Step 4: Pipeline Summary")
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    print(f"\n📅 Pipeline started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📅 Pipeline ended: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"⏱️  Total duration: {duration:.1f} seconds")
    
    print("\n📁 Generated Files:")
    print("   data/raw/fake_news.csv                 - Main dataset")
    print("   data/raw/fake_news_sample.csv          - Sample dataset")
    print("   data/raw/fake_news_data_dictionary.json - Data dictionary")
    print("   data/processed/analysis/               - Analysis results")
    print("   data/processed/unified/                - Unified data & splits")
    
    print("\n📋 Next Steps:")
    print("   1. Review analysis report: data/processed/analysis/analysis_report.md")
    print("   2. Check processed splits: data/processed/unified/")
    print("   3. Proceed to model training with the processed data")
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETE - READY FOR MODEL TRAINING")
    print("="*60 + "\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run fake news data pipeline")
    parser.add_argument("--generate", action="store_true", 
                       help="Generate sample data first")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run pipeline
    run_complete_fake_news_pipeline(generate_data=args.generate)
