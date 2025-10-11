"""Example to check the number of messages in an MCAP file."""

import time

from mcap.reader import NonSeekingReader

mcap_file = "/home/daniel/Downloads/episode_0000_0.mcap"


def get_message_count(file_path: str) -> int:
    """Get the total number of messages in an MCAP file."""
    with open(file_path, "rb") as f:
        reader = NonSeekingReader(f)
        summary = reader.get_summary()
        if summary is None:
            raise ValueError("No summary found in MCAP file")
    return summary.statistics.message_count if summary.statistics else 0


start_time = time.time()
message_count = get_message_count(mcap_file)
end_time = time.time()
print(f"Message count: {message_count}")
print(f"Time taken: {end_time - start_time:.4f} seconds")
