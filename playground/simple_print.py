import json
import linecache

corpus_file = "/data/users/cluo86/FlashRAG/corpus/wiki18_100w.jsonl"

def get_contents_by_ids(ids):
    """Get content for a list of IDs using direct line access"""
    contents = {}
    
    for id in ids:
        # IDs start from 0, but linecache is 1-indexed
        line = linecache.getline(corpus_file, id + 1)
        if line.strip():
            try:
                content = json.loads(line)
                contents[id] = content
            except json.JSONDecodeError:
                contents[id] = None
    
    return contents

# Read the pipeline pool IDs from a file
def read_pipeline_ids(pool_file="pipeline_pool.txt"):
    pipeline_ids = []
    with open(pool_file, 'r') as f:
        for line in f:
            if ":" in line:
                # Extract IDs after the colon
                ids_part = line.split(":", 1)[1].strip()
                if ids_part:
                    # Convert comma-separated string to integers
                    ids = [int(id.strip()) for id in ids_part.split(",")]
                    pipeline_ids.extend(ids)
    return pipeline_ids

# Example usage
if __name__ == "__main__":
    # Get IDs from pipeline pool or use a test list
    try:
        ids = read_pipeline_ids()
        if not ids:
            ids = [0, 10, 100, 1000]  # Example IDs if no file found
    except FileNotFoundError:
        ids = [0, 10, 100, 1000]  # Example IDs
        
    ids = [101436,101434,74400,101435,74401,101432,130187,74402,74562,101437]
    
    print(f"Retrieving content for {len(ids)} IDs...")
    contents = get_contents_by_ids(ids)
    
    # Print the contents
    for id, content in contents.items():
        if content:
            # Print ID and a preview of the content
            print(f"ID {id}: {json.dumps(content)[:100]}...")
        else:
            print(f"ID {id}: Content not found or invalid")