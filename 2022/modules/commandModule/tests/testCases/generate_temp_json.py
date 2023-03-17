import random

valid_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890"


def generate_id() -> str:
    return ''.join(random.choice(valid_letters) for _ in range(6))


def generate_temp_json(filepath: str) -> str:
    with open(filepath, "r") as original_file:
        new_file_name = generate_id() + ".json"
        with open(new_file_name, "w") as new_file:
            for line in original_file:
                new_file.write(line)
                new_file.write('\n')
        return new_file_name
