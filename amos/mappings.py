import numpy as np
"""
gpt-5
text.format: text
effort: medium
verbosity: medium
summary: auto
26.8
Prompt by GV: 
create a python function to return all unique combinations of the first 4 columns of nmat as an array of size N by 4. For example nmat array([[12, 'Violoncello', 0, 45, 0.0, 0.5, 52, 32],

[13, 'Contrabass', 1, 45, 0.0, 0.5, 40, 32],

[9, 'Violin I', 13, 45, 0.5, 0.5, 64, 33],

[10, 'Violin II', 14, 45, 0.5, 0.5, 59, 33],

[11, 'Viola', 15, 45, 0.5, 0.5, 55, 34],

[12, 'Violoncello', 0, 45, 1.0, 0.5, 52, 31],

[13, 'Contrabass', 1, 45, 1.0, 0.5, 40, 30],

[9, 'Violin I', 13, 45, 1.5, 0.5, 66, 32],

[10, 'Violin II', 14, 45, 1.5, 0.5, 60, 30],

[11, 'Viola', 15, 45, 1.5, 0.5, 57, 36]], dtype=object)
"""
def learn_quaterna_mapping(nmat,ytarget="track-channel"):
    """
    Code generated with https://platform.openai.com/
    Model gpt-4.1, text.format: text, temp: 1.00, tokens: 2048, top_p: 1.00, store: true
    Prompt by Gissel Velarde
    Prompt:
    the first 4 columns of a an numpy array with 8 columns always appear in quaterna. Write a function that learns the quaterna, given that column 3 is used as a label for a machine learning model. Then, once the model predicts the label for column 4, fill the corresponding values for columns 0, 1 and 2.
    """
    if ytarget=="program":
        """
        When target is program
        Learns the mapping from label in column 3 to columns 0, 1, 2.
        Returns a dictionary: label_val -> [col0, col1, col2]
        """
        mapping = {}
        for row in nmat:
            label = row[3]
            # Only keep the last if duplications appear (can add check for consistency)
            mapping[label] = row[:3].tolist()
        #return mapping
    else:
        """
        When target is track-channel
        Returns all unique combinations of the first 4 columns of a NumPy array,
        sorted by the first and third columns.
    
        Args:
            nmat (np.ndarray): A NumPy array with at least 4 columns.
    
        Returns:
            np.ndarray: A new NumPy array containing only the unique rows
                        from the first 4 columns of the input array, sorted.
        """
        # Extract the first 4 columns
        first_four_columns = nmat[:, :4]
    
        # Convert each row (list) into a tuple so it can be added to a set
        unique_rows_as_tuples = set(map(tuple, first_four_columns))
    
        # Convert the set of tuples back to a list of lists and sort it.
        unique_rows = sorted(list(unique_rows_as_tuples), key=lambda x: (x[0], x[2]))
        mapping = np.array(unique_rows, dtype=object)
    # Convert the sorted list back to a NumPy array
    return mapping


#26.8
#Gemini
"""
Prompt GV.
Given y which has the first and third column of map_array, create a function that returns nmat_L, if a combination of first, second, and third column repeats with a different number in the fourth column, insert in nmat_L, only the first combination of the first 3 columns. 

For example:
y =  ['1_1', '1_1', '2_1', '3_1', '4_2', '4_2']    
          
 map_array = np.array([[1, 'V',1,70],
                          [2, 'B',1,72],
                          [3, 'F',1,73],
                          [4, 'F',2,73]]
                          [3, 'F',1,98],
                          [4, 'F',2,98]], dtype=object)
                          
                          
nmat_L = np.array([[1,'V' ,1,70],
                           [1,'V', 1,70],
                           [2, 'B',1,72],
                           [3,'F', 1,73],
                           [4, 'F',2,73],
                           [4,'F', 2,73]], dtype=object)
"""

def fill_quaterna_columns(y, map_array, ytarget="track-channel"):
    if ytarget=="program":
        """
        Code generated with https://platform.openai.com/
        Model gpt-4.1, text.format: text, temp: 1.00, tokens: 2048, top_p: 1.00, store: true
        Prompt by Gissel Velarde
        Prompt:
        the first 4 columns of a an numpy array with 8 columns always appear in quaterna. Write a function that learns the quaterna, given that column 3 is used as a label for a machine learning model. Then, once the model predicts the label for column 4, fill the corresponding values for columns 0, 1 and 2.
        """
        """
        Given a list/array of predicted labels, use the mapping to reconstruct cols 0, 1, 2.
        Returns an array of shape (N, 3) where N is the number of predicted labels.
        """
        filled = []
        for label in y:
            if label in map_array:
                filled.append(map_array[label])
            else:
                # handle unknown labels (e.g., with np.nan)
                filled.append([np.nan, np.nan, np.nan])
        nmat_L_list = np.concatenate((np.array(filled), y.reshape(-1, 1)), axis=1)
        #return np.array(filled)
    else:
        """
        Generates a NumPy array (nmat_L) from a list of strings (y) by mapping
        the strings to full rows from a map array. The key for the map is formed
        from the first and third columns of map_array. If a combination of the
        first, second, and third columns repeats, only the first occurrence is used.
    
        Args:
            y (list): A list of strings, where each string represents a key for a
                      row in the map_array (e.g., '1_1').
            map_array (np.ndarray): A 2D NumPy array containing the data to map to.
    
        Returns:
            np.ndarray: A new NumPy array (nmat_L) containing the full rows
                        from the map_array.
        """
        # Create a dictionary for fast lookups. We only store the first
        # combination of the first three columns.
        row_map = {}
        for row in map_array:
            key = f"{row[0]}_{row[2]}"
            if key not in row_map:
                row_map[key] = row
        
        # Use a list comprehension to build the new array based on the strings in y
        nmat_L_list = [row_map[key] for key in y]
        nmat_L_list =  np.array(nmat_L_list, dtype=object)
    # Convert the list of arrays into a single NumPy array
    return nmat_L_list