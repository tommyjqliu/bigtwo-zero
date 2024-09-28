def append_to_log(file_path, log_message):
    try:
        with open(file_path, "a") as log_file:
            log_file.write(log_message + "\n")
        print(f"Appended to {file_path}: {log_message}")
    except IOError as e:
        print(f"Error appending to {file_path}: {e}")
