# -*- coding: utf-8 -*-
import os
import socket
import sys
import time


def main():
    cache_dir, nnodes, nproc_per_node, node_rank, *args = sys.argv[1:]

    nnodes = int(nnodes)
    nproc_per_node = int(nproc_per_node)
    node_rank = int(node_rank) - 1

    rank_file = os.path.join(cache_dir, "rank." + str(node_rank))

    with open(rank_file, "w") as f:
        f.write(socket.gethostname())

    timeout = 60

    while timeout > 0:
        timeout -= 5
        time.sleep(5)

        if len(os.listdir(cache_dir)) == nnodes:
            break

    # Read master info
    with open(os.path.join(cache_dir, "rank.0"), "r") as f:
        master_address = f.read().strip()
        print('Master address: ', master_address)

        os.system(
            "python -m torch.distributed.launch --nproc_per_node={} --nnodes={} --node_rank={} --master_addr={} --master_port=9000 {}".format(
                nproc_per_node, nnodes, node_rank, master_address, " ".join(args)
            )
        )


if __name__ == "__main__":
    main()
