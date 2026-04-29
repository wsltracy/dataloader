import numpy as np
from PIL import Image
from streaming import MDSWriter

# Local or remote directory in which to store the compressed output files
data_dir = './datas/stream/1'

# A dictionary mapping input fields to their data types
columns = {
    'image': 'jpeg',
    'class': 'int'
}

# Shard compression, if any
compression = 'zstd'

# Save the samples as shards using MDSWriter
with MDSWriter(out=data_dir, columns=columns, compression=compression) as out:
    for i in range(10000):
        sample = {
            'image': Image.fromarray(np.random.randint(0, 256, (32, 32, 3), np.uint8)),
            'class': np.random.randint(10),
        }
        out.write(sample)