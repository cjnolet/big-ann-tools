import numpy as np


def load_chunk(file_name, chunk, chunk_size, dt):
  dt = np.dtype(args.dt)

  in_file = open(args.file, "rb")

  n_pts = int.from_bytes(in_file.read(4), byteorder='little', signed=False)
  n_dim = int.from_bytes(in_file.read(4), byteorder='little', signed=False)

  chunk = args.chunk # TODO: Make this an argument to the script
  chunk_size = args.chunk_size # TODO: Make this an argument to the script
  start_bytes = 8

  entry_bytes_start = (n_dim * dt.itemsize * chunk_size * chunk) + start_bytes

  in_file.seek(entry_bytes_start)

  chunk_size = min(chunk_size, n_pts - (chunk * chunk_size))

  b = in_file.read(chunk_size * dt.itemsize * n_dim)

  return np.frombuffer(b, dtype=dt, count=chunk_size * n_dim).reshape((chunk_size, n_dim))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("file", type=str,
                    help="File to load a chunk from")
    ap.add_argument("chunk", type=int,
                    help="Chunk to load")
    ap.add_argument("chunk_size", type=int,
                    help="Number of rows to load from file")
    ap.add_argument("dt", type=str,
                    help="Datatype of vectors")
    args = ap.parse_args()

    arr = load_chunk(args.file, args.chunk, args.chunk_size, args.dt)

    print(str(arr))
