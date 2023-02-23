import numpy as np


def decompose_2d(data, num_of_chunks):
    chunk_size_y = int(data.shape[0] / num_of_chunks)
    chunk_size_x = int(data.shape[1] / num_of_chunks)
    splitted_data = np.zeros(
        shape=(num_of_chunks * num_of_chunks, chunk_size_y, chunk_size_x)
    )
    for j in range(num_of_chunks):
        for i in range(num_of_chunks):
            splitted_data[i + num_of_chunks * j] = np.array(
                data[
                    j * (chunk_size_y) : (j + 1) * chunk_size_y,
                    i * (chunk_size_x) : (i + 1) * chunk_size_x,
                ]
            )

    return splitted_data


def recompose_2d(splitted_data, num_of_chunks):
    height, width = splitted_data[0].shape
    recomposed_data = np.zeros(
        shape=(height * num_of_chunks, width * num_of_chunks)
    )
    for j in range(num_of_chunks):
        for i in range(num_of_chunks):
            recomposed_data[
                j * (height) : (j + 1) * height,
                i * (width) : (i + 1) * width,
            ] = splitted_data[i + num_of_chunks * j]

    return recomposed_data
