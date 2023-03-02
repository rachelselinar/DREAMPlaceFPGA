import os
import sys 
import time
import argparse


def read_io_site(io_file):
    
    inst_to_site = {}
    with open(io_file, 'r') as fin:
        for line in fin:
            inst, site_name = line.split()
            if '/' in inst:
                inst = inst.split('/')[0]
                
            inst_to_site[inst] = site_name

    x_max = 206
    y_max = 300
    z_max = 64

    return inst_to_site


def generate_io_mapping():

    io_sitemap = {}
  
    z_indices = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 1, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 0]
    x_indices = [68, 137]

    x_max = 206
    y_max = 300
    z_max = 64
    
    site_x = 0
    site_y = 0

    # IOB sites
    for x in x_indices:
        for y in range(0, int(y_max/30)):
            for z in z_indices:
                key = (x, y*30, z)
                io_sitemap[key] = 'IOB_X' + str(site_x) + 'Y' + str(site_y)
                site_y += 1

        site_x += 1
        site_y = 0


    # BUFGCE sites
    buf_z_max = 120
    buf_x_indices = [69, 138]

    buf_x = 0
    buf_y = 0
    for x in buf_x_indices:
        for z in range(0, buf_z_max):
            key = (x, 0, z)
            io_sitemap[key] = 'BUFGCE_X' + str(buf_x) + 'Y' + str(buf_y)
            buf_y += 1

        buf_x += 1
        buf_y = 0


    return io_sitemap


if __name__ == "__main__":
    """ Main function converting io_placement.txt to design.pl """

    parser = argparse.ArgumentParser()
    parser.add_argument("--io_place")
    args = parser.parse_args()


    inst_to_site = read_io_site(args.io_place)
    sitemap = generate_io_mapping()

    with open('design.pl', 'w') as pl_file:
        for key, value in sitemap.items():
            x, y, z = key
            for inst, sitename in inst_to_site.items():
                if sitename == value:
                    line = inst + ' ' + str(x) + ' ' + str(y) + ' ' + str(z) + ' FIXED'
                    pl_file.write(line + os.linesep)
  
