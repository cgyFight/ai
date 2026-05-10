def read_data(file_path) -> str:
    """Reads data from a file and returns its content."""
    with open(file_path, 'r') as file:
        data = file.read()
    return data

def split_into_chunks(data: str) -> list[str]:
    """Splits the input data into chunks based on double newlines."""
    chunks = data.split('\n\n')
    return [chunk.strip() for chunk in chunks if chunk.strip()]

def combine_chunks_when_start_with(chunks: list[str], start_str: str) -> list[str]:
    """if a chunk starts with start_str, mean it is a new section, so it should be combined with next chunk."""
    combined_chunks = []
    skip_next = False

    for i in range(len(chunks)):
        if skip_next:
            skip_next = False
            continue

        current_chunk = chunks[i]
        if current_chunk.startswith(start_str) and i + 1 < len(chunks):
            combined_chunk = current_chunk + "\n\n" + chunks[i + 1]
            combined_chunks.append(combined_chunk)
            skip_next = True
        else:
            combined_chunks.append(current_chunk)

    return combined_chunks


if __name__ == "__main__":
    file_path = "data.md"
    chunks = combine_chunks_when_start_with(
        split_into_chunks(read_data(file_path)),
        start_str="#")
    for c in chunks:
        print("---- Chunk ----")
        print(c)
        print("----------------\n")