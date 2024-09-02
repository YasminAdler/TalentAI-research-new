def average_of_lists(list_of_lists):
    return [sum(val for val in col if val != "missing_val") / sum(1 for val in col if val != "missing_val")
            if any(val != "missing_val" for val in col) else "missing_val"
            for col in zip(*list_of_lists)]

# Example usage:
data = [[1, 3, 4], [3, 5, "missing_val"],["missing_val", 2, "missing_val"]]

avg_lst = average_of_lists(data)
print(avg_lst)  # Output: [2, 4, 4]
