"""Implementations on the Abelian Sandpile and Conway's
Game of life on various meshes"""
import numpy as np


def sandpile(initial_state, periodic=False):
    """
    Generate final state for Abelian Sandpile.
    Parameters
    ----------
    initial_state : array_like or list of lists
        Initial 2d state of grid in an array of integers.
    Returns
    -------
    numpy.ndarray
         Final state of grid in array of integers.
    """

    # Set the maximum grain for a cell
    maxi = 4
    # Counting...
    i = 0

    # Produce a 'sink' border
    def false_border(mesh):
        '''
        Create a border filled with zero for the mesh
        :param mesh: the input mesh with random integers from 0-6
        :param n: the size of the square mesh
        :return: the square mesh with borders of zeros
        '''
        mesh[0, :] = False
        mesh[-1, :] = False
        mesh[:, 0] = False
        mesh[:, -1] = False
        return mesh

    def periodic_border(mesh):
        mesh[1:-1, 0] = mesh[1:-1, -2]
        mesh[1:-1, -1] = mesh[1:-1, 1]
        mesh[0, 1:-1] = mesh[-2, 1:-1]
        mesh[-1, 1:-1] = mesh[1, 1:-1]
        return mesh

    mesh = np.pad(initial_state, pad_width=1, mode='constant', constant_values=0)

    while np.max(mesh) >= maxi:

        # call the zero_border function
        if periodic:
            periodic_border(mesh)
        else:
            false_border(mesh)

        # Find cells larger than 4
        highest = mesh >= maxi
        mesh[highest] -= maxi
        row, col = np.where(highest)[0], np.where(highest)[1]

        # The large pile gives grain to cells surrounding it
        right = (row+1, col)
        left = (row-1, col)
        abo = (row, col+1)
        bot = (row, col-1)
        mesh[right] += int(maxi/4)
        mesh[left] += int(maxi/4)
        mesh[bot] += int(maxi/4)
        mesh[abo] += int(maxi/4)

        i += 1
    return mesh[1:-1, 1:-1]
    # raise NotImplementedError
    # raise NotImplementedError


def life(initial_state, nsteps, periodic=False):
    """
    Perform iterations of Conway’s Game of Life.
    Parameters
    ----------
    initial_state : array_like or list of lists
        Initial 2d state of grid in an array of booleans.
    nsteps : int
        Number of steps of Life to perform.
    periodic : bool
        If true, then grid is assumed periodic.
    Returns
    -------
    numpy.ndarray
         Final state of grid in array of booleans
    """

    def false_border(mesh):
        '''
        Create a border filled with zero for the mesh
        :param mesh: the input mesh with random integers from 0-6
        :param n: the size of the square mesh
            :return: the square mesh with an outer boundary filled with False
        '''
        mesh[0, :] = False
        mesh[-1, :] = False
        mesh[:, 0] = False
        mesh[:, -1] = False
        return mesh

    def periodic_border(mesh):
        mesh[1:-1, 0] = mesh[1:-1, -2]
        mesh[1:-1, -1] = mesh[1:-1, 1]
        mesh[0, 1:-1] = mesh[-2, 1:-1]
        mesh[-1, 1:-1] = mesh[1, 1:-1]
        return mesh

    def counter(i, j, mesh):
        '''
        Count the existence of neighbouring cells with True value
        :param i: The index of rows from condition
                cell value == True(row_true) or False(row_false)
        :param j: The index of colums from condition
                cell value == True(col_true) or False(col_false)
        :param mesh: The copied matrix
        :return: The sum of True neighbours
        '''

        left = (mesh[i-1, j]).astype(int)
        lefttop = mesh[i-1, j-1]
        top = mesh[i, j-1]
        topright = mesh[i+1, j-1]
        right = mesh[i+1, j]
        rightbot = mesh[i+1, j+1]
        bot = mesh[i, j+1]
        botleft = mesh[i-1, j+1]

        return left + lefttop + top + topright + right + rightbot + bot + botleft


    mesh = np.pad(initial_state, pad_width=1, mode='constant', constant_values=False)
    i = 0
    while i < nsteps:

        if periodic:
            mesh = false_border(mesh)
        else:
            mesh = periodic_border(mesh)

        mesh_true = mesh.copy()
        row_true = np.where(mesh_true[1:-1, 1:-1] == 1)[0] + 1
        col_true = np.where(mesh_true[1:-1, 1:-1] == 1)[1] + 1
        mesh_false = mesh.copy()
        row_false = np.where(mesh_false[1:-1, 1:-1] == 0)[0] + 1
        col_false = np.where(mesh_false[1:-1, 1:-1] == 0)[1] + 1

        sum_true = counter(row_true, col_true, mesh_true)
        sum_true[(sum_true != 2) & (sum_true != 3)] = False
        sum_true[(sum_true == 2) | (sum_true == 3)] = True

        sum_false = counter(row_false, col_false, mesh_false)
        sum_false[sum_false != 3] = False
        sum_false[sum_false == 3] = True

        mesh[row_true, col_true], mesh[row_false, col_false] = sum_true, sum_false
        i += 1
    return mesh[1:-1, 1:-1]
    # raise NotImplementedError


def lifetri(initial_state, nsteps):
    """
    Perform iterations of Conway’s Game of Life on a triangular tessellation.
    Parameters
    ----------
    initial_state : array_like or list of lists
        Initial state of grid on triangles.
    nsteps : int
        Number of steps of Life to perform.
    periodic : bool, optional
        If true, then grid is assumed periodic.
    Returns
    -------
    array
    Final state of tessellation.
    """
    def false_border(mesh):
        '''
        Create a border filled with zero for the mesh
        :param mesh: the input mesh with random integers from 0-6
        :param n: the size of the square mesh
            :return: the square mesh with an outer boundary filled with False
        '''
        mesh[0, :] = False
        mesh[-1, :] = False
        mesh[:, 0] = False
        mesh[:, -1] = False
        return mesh

    def give_status(i, j, mesh):
        mesh = mesh.astype(str)
        cor = np.stack((i, j), axis=1)
        cor_check = (cor[:, 0]+cor[:, 1])%2
        ieven = np.where(cor_check == 0)
        mesh[cor[ieven][:, 0], cor[ieven][:, 1]] = 'Up'
        iodd = np.where(cor_check == 1)
        mesh[cor[iodd][:, 0], cor[iodd][:, 1]] = 'Down'

        return mesh

    def up_counter(i, j, mesh):
        '''
        Count the existence of neighbouring cells with True value
        :param i: The index of rows from condition
                cell value == True(row_true) or False(row_false)
        :param j: The index of colums from condition
                cell value == True(col_true) or False(col_false)
        :param mesh: The copied matrix
        :return: The sum of True neighbours
        '''
        left = (mesh[i, j-1]).astype(int)
        lefttop = mesh[i-1, j-1]
        left2 = mesh[i, j-2]
        top = mesh[i-1, j]
        topright = mesh[i-1, j+1]
        right = mesh[i, j+1]
        right2 = mesh[i, j+2]
        rightbot2 = mesh[i+1, j+2]
        rightbot = mesh[i+1, j+1]
        bot = mesh[i+1, j]
        botleft = mesh[i+1, j-1]
        botleft2 = mesh[i+1, j-2]
        sum_neighbour = (left + lefttop + left2 + top + topright + right + right2 +
                         rightbot + rightbot2 + bot + botleft + botleft2)
        return sum_neighbour

    def down_counter(i, j, mesh):
        left = (mesh[i, j-1]).astype(int)
        left2 = mesh[i, j-2]
        lefttop2 = mesh[i-1, j-2]
        lefttop = mesh[i-1, j-1]
        top = mesh[i-1, j]
        topright = mesh[i-1, j+1]
        topright2 = mesh[i-1, j+2]
        right2 = mesh[i, j+2]
        right = mesh[i, j+1]
        rightbot = mesh[i+1, j+1]
        bot = mesh[i+1, j]
        botleft = mesh[i+1, j-1]
        return (left + left2 + lefttop2 + lefttop + top + topright +
                topright2 + right2 + right + rightbot + bot + botleft)

    def check_rule(test_ary, env, fert, status='life'):
        if status == 'life':
            return np.isin(test_ary, env)
        return np.isin(test_ary, fert)

    pad_width = 2
    env = (4, 5, 6)
    fert = 4

    mesh = np.pad(initial_state, pad_width=pad_width, mode='constant', constant_values=False)

    i = 0
    while i < nsteps:

        false_border(mesh)
        mesh_true = mesh.copy()
        row_true, col_true = np.where(mesh_true == 1)
        mesh_false = mesh.copy()
        row_false = np.where(mesh_false[2:-2, 2:-2] == 0)[0]+2
        col_false = np.where(mesh_false[2:-2, 2:-2] == 0)[1]+2

        # Give status of a copied matrix
        mesh_true_int = give_status(row_true, col_true, mesh_true)
        # Find the index of Up and Down triangles
        row_true_up, col_true_up = np.where(mesh_true_int == 'Up')
        row_true_down, col_true_down = np.where(mesh_true_int == 'Down')
        # Count for Up and Down cells separately
        sum_true_up = up_counter(row_true_up, col_true_up, mesh_true)
        sum_true_down = down_counter(row_true_down, col_true_down, mesh_true)
        # Replace the sum of neighbouring values depending on the rule
        true_up_new = check_rule(sum_true_up, env, fert, status='life')
        true_down_new = check_rule(sum_true_down, env, fert, status='life')
        mesh[row_true_up, col_true_up] = true_up_new
        mesh[row_true_down, col_true_down] = true_down_new

        mesh_false_int = give_status(row_false, col_false, mesh_false)
        row_false_up, col_false_up = np.where(mesh_false_int == 'Up')
        row_false_down, col_false_down = np.where(mesh_false_int == 'Down')
        sum_false_up = up_counter(row_false_up, col_false_up, mesh_false)
        sum_false_down = down_counter(row_false_down, col_false_down, mesh_false)
        # Replace the sum of neighbouring values depending on the rule
        false_up_new = check_rule(sum_false_up, env, fert, status='dead')
        false_down_new = check_rule(sum_false_down, env, fert, status='dead')
        mesh[row_false_up, col_false_up] = false_up_new
        mesh[row_false_down, col_false_down] = false_down_new

        i += 1
    return mesh[2:-2, 2:-2]


def life_generic(matrix, initial_state, nsteps, environment, fertility):
    """
    Perform iterations of Conway’s Game of Life for an arbitrary
    collection of cells.
    Parameters
    ----------
    matrix : 2d array of bools
        a boolean matrix indicating neighbours for each row
    initial_state : 1d array of bools
        Initial state vectr.
    nsteps : int
        Number of steps of Life to perform.
    environment : set of ints
        neighbour counts for which live cells survive.
    fertility : set of ints
        neighbour counts for which dead cells turn on.
    Returns
    -------
    array
         Final state.
    """
    mesh = initial_state
    i = 0
    while i < nsteps:
        # Multiplication simply gives the valid neighbours i.e. value == True
        sum_array = np.sum(matrix * mesh, axis=1)
        # Get indexes of True/False values in 1d array
        row_true = np.where(mesh == 1)
        row_false = np.where(mesh == 0)

        # Get the sum array for True/False value separately and check the rules
        sum_true, sum_false = sum_array[row_true], sum_array[row_false]

        mesh[row_true] = np.isin(sum_true, list(environment))
        mesh[row_false] = np.isin(sum_false, list(fertility))

        i += 1
    return mesh
