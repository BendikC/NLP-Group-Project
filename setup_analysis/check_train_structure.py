import json
import os
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

def check_train_structure():
    """
    Checks and validates the structure of the training data in data/qa/train.json
    Logs statistics and any potential issues found.
    """
    train_path = "data/qa/train.json"
    
    if not os.path.exists(train_path):
        logger.error(f"Training file not found at {train_path}")
        return False
        
    try:
        with open(train_path, "r") as f:
            train_data = json.load(f)
            
        if not isinstance(train_data, list):
            logger.error("Training data must be a list of examples")
            return False

        # Collect statistics
        stats = defaultdict(int)
        required_fields = {'_id', 'type', 'question', 'context', 'supporting_facts', 'evidences', 'answer'}
        
        for idx, item in enumerate(train_data):
            # Check required fields
            missing_fields = required_fields - set(item.keys())
            if missing_fields:
                logger.error(f"Entry {idx}: Missing required fields: {missing_fields}")
                return False
            
            # Validate types
            if not isinstance(item['context'], list):
                logger.error(f"Entry {idx}: 'context' must be a list")
                return False
                
            if not isinstance(item['supporting_facts'], list):
                logger.error(f"Entry {idx}: 'supporting_facts' must be a list")
                return False
                
            if not isinstance(item['evidences'], list):
                logger.error(f"Entry {idx}: 'evidences' must be a list")
                return False
                
            # Collect statistics
            stats['total_entries'] += 1
            stats[f"type_{item['type']}"] += 1
            stats['total_context_entries'] += len(item['context'])
            stats['total_supporting_facts'] += len(item['supporting_facts'])
            stats['total_evidences'] += len(item['evidences'])
            
        # Log statistics
        logger.info("\nDataset Statistics:")
        for key, value in stats.items():
            logger.info(f"{key}: {value}")
            
        return True
        
    except json.JSONDecodeError:
        logger.error(f"Failed to parse {train_path} as JSON")
        return False
    except Exception as e:
        logger.error(f"Error checking training data: {str(e)}")
        return False

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    check_train_structure()
