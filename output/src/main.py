#!/usr/bin/env python3
"""
Main entry point for the application.
"""

import sys
import logging
from src.core.config import Config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main function to run the application."""
    logger.info("Starting application")
    
    # Load configuration
    try:
        config = Config()
        logger.info("Configuration loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        sys.exit(1)
    
    # Application logic would go here
    logger.info("Application running with configuration: %s", config.get_config())
    
    return 0

if __name__ == "__main__":
    sys.exit(main())