#!/usr/bin/env python

import sys, os

sys.path.append(os.getcwd())

def main():
    assert sys.argv[1] in ('naming', 'Post')
    prog = sys.argv.pop(1)
    if prog == 'naming':
        from BSLSolver.common import naming
    #elif prog == 'Post': not used yet...
    #    from BSLSolver import Post
    else:
        raise NotImplementedError
if __name__ == '__main__':
    main()
