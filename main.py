import os
import sys
import logging
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main() -> None:
    """
    Main function that runs the application
    """
    try:
        logger.info("Starting application...")
        
        # Your main application logic here
        print("Hello from main.py!")
        
        # Example: Check if requirements.txt exists
        if os.path.exists("requirements.txt"):
            logger.info("requirements.txt found")
            with open("requirements.txt", "r") as f:
                requirements = f.read()
                print(f"\nDependencies in requirements.txt:\n{requirements}")
        else:
            logger.warning("requirements.txt not found")
        
        logger.info("Application completed successfully!")
        
    except Exception as e:
        logger.error(f"Application failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
