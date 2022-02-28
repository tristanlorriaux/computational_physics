# Import standard libraries
import numpy as np
from scipy.special import gamma
from scipy import sparse
import matplotlib.pyplot as plt
from matplotlib import animation, cm, colors, rc
import random
import time
from imp import reload
rc('animation', html='html5')

# Import library of our own functions
import MPCMolecularDynamics as MD
reload(MD)

######################################
########## COMPRESSIBILITY ###########
######################################

#######A routine for generating or extending individual NVT-MD trajectories for a LJ systems########


def Generate_LJ_NVT_MolecularDynamics_Trajectory(d,m,LBox,kT,run_time,
                                                 starting_configuration=[],
                                                 time_step = 0.01,
                                                 number_of_time_steps_between_stored_configurations=100,
                                                 number_of_time_steps_between_velocity_resets=100,
                                                 debug=False):
    """
    generates a NVT MD simulations of a LJ system with sigma=epsilon=1  
        - where the particle masses are specified in the array m
        - so that NParticles = m.size
        - in a volume V=(LBox,LBox) at a specified temperature kT
        - with a time step of time_step tau 
          where the LJ unit of time is calculated as a function of m[-1], i.e. the mass of the LAST particle
        - runs are either started from 
                a specified starting configuration [t,x,y,vx,vy] or
                initialized with zero velocities and particles placed on a square grid
        - the simulations are thermostated by redrawing random velocities from the 
          Maxwell-Boltzmann distribution number_of_time_steps_between_velocity_resets time steps
        - the function returns 
                trajectory lists t_tr, x_tr, y_tr, vx_tr, vy_tr, uPot_tr, uKin_tr, pPot_tr, pKin_tr 
                     of sampling times and sampled coordinates, velocities and energies and pressures
                a final configuration [t,x,y,vx,vy] from which the run can be restarted
                while the energies and pressures are recorded at every time step, configurations 
                     and velocities are stored at a time interval of time_between_stored_configurations
                    
    """

    NParticles = m.size
    sigma = 1
    epsilon = 1
    #unit of time
    tau = sigma*np.sqrt(m[-1]/epsilon)      

    # define the length of the trajectory
    number_of_timesteps = int(np.round(run_time/time_step))

    #starting configuration
    if starting_configuration!=[]:
        [t,x,y,vx,vy] = starting_configuration
    else:
        # default initial state
        x,y = MD.GridPositionsIn2d(LBox,LBox,NParticles)
        vx = MD.RandomVelocities(m,kT)
        vy = MD.RandomVelocities(m,kT)
        t = 0
        if debug:
            print("No starting configuration")

    #initialize Trajectory
    t_tr = []
    x_tr = []
    vx_tr = []
    y_tr = []
    vy_tr = []

    fx,fy = MD.LJ_forces_as_a_function_of_positions(d,epsilon,sigma,LBox,(x,y))
    # force for initial configuration needed for first time step

    for timestep in range(number_of_timesteps):
        (x,y),(vx,vy) = MD.VelocityVerletTimeStepPartOne(m,(x,y),(vx,vy),(fx,fy),time_step)
        fx,fy = MD.LJ_forces_as_a_function_of_positions(2,epsilon,sigma,LBox,(x,y))
        (x,y),(vx,vy) = MD.VelocityVerletTimeStepPartTwo(m,(x,y),(vx,vy),(fx,fy),time_step)
        t += time_step
        
        t_tr.append(t)
        x_tr.append(x)
        vx_tr.append(vx)
        y_tr.append(y)
        vy_tr.append(vy)
    
        # thermostat: reinitialise velocities to control temperature
#        if np.mod( timestep*time_step, time_between_velocity_resets ) == 0.0 and timestep>1:
        if timestep%number_of_time_steps_between_velocity_resets == 0 and timestep>1:
            vx = MD.RandomVelocities(m,kT)
            vy = MD.RandomVelocities(m,kT)

    # convert trajectory lists to arrays to simplify the data analysis
    t_tr = np.array(t_tr)
    x_tr = np.array(x_tr)
    vx_tr = np.array(vx_tr)
    y_tr = np.array(y_tr)
    vy_tr = np.array(vy_tr)

    # analyse results 
    uPot_tr = MD.LJ_energy_as_a_function_of_positions(d,epsilon,sigma,LBox,(x_tr,y_tr))
    uKin_tr = MD.TotalKineticEnergy(m,vx_tr) + MD.TotalKineticEnergy(m,vy_tr)
    pPot_tr = MD.LJ_virial_pressure_as_a_function_of_positions(d,epsilon,sigma,LBox,(x_tr,y_tr)) 
    pKin_tr = MD.KineticPressure_as_a_function_of_velocities(d,LBox,m,(vx_tr,vy_tr))
    
    # reduce the number of stored configurations and velocities
#    skip = int(time_between_stored_configurations / delta_t)
    skip = number_of_time_steps_between_stored_configurations
    x_tr = x_tr[::skip]
    y_tr = y_tr[::skip]
    vx_tr = vx_tr[::skip]
    vy_tr = vy_tr[::skip]    
    # note that t_tr is not compressed as it contains the times corresponding to the stored energies and pressures
    # as a consequence a corresponding skipping operation needs to be performed, when configurations are plotted 
    # as a function of time
    
    return t_tr, x_tr, y_tr, vx_tr, vy_tr, uPot_tr, uKin_tr, pPot_tr, pKin_tr, [t,x,y,vx,vy]


def Generate_Ensemble_of_LJ_NVT_MolecularDynamics_Trajectories(d,m,LBox,kT,NTrajectories,run_time,
                                                               list_of_starting_configurations=[],
                                                               time_step=0.01,
                                                               number_of_time_steps_between_stored_configurations=100,
                                                               number_of_time_steps_between_velocity_resets=100,
                                                               debug=False):
    """
    uses Generate_LJ_NVT_MolecularDynamics_Trajectory to

    generate an ensemble of NTrajectories NVT MD simulations of a LJ system with sigma=epsilon=1  
        - where the particle masses are specified in the array m
        - so that NParticles = m.size
        - in a volume V=(LBox,LBox) at a specified temperature kT
        - with a time step of time_step tau 
          where the LJ unit of time is calculated as a function of m[-1], i.e. the mass of the LAST particle
        - runs are either started from 
                a list of specified starting configuration [[t,x,y,vx,vy], ...] or
                initialized with zero velocities and particles placed on a square grid
        - the simulations are thermostated by redrawing random velocities from the 
          Maxwell-Boltzmann distribution at intervals of time_between_velocity_resets
        - the function returns 
                trajectory ensemble lists t_tr_ens, x_tr_ens, y_tr_ens, vx_tr_ens, vy_tr_ens, uPot_tr_ens, uKin_tr_ens, pPot_tr_ens, pKin_tr_ens 
                     of sampling times and sampled coordinates, velocities and energies and pressures
                a list of final configurations [[t,x,y,vx,vy], ...] from which the runs can be restarted
                while the energies and pressures are recorded at every time step, configurations 
                     and velocities are stored at a time interval of time_between_stored_configurations      
    """
    # initialize lists to collect ENSEMBLES of trajectories
    t_tr_ens = []
    x_tr_ens = []
    vx_tr_ens = []
    y_tr_ens = []
    vy_tr_ens = []
    uKin_tr_ens = []
    uPot_tr_ens = []
    pKin_tr_ens = []
    pPot_tr_ens = []
    
    # convert empty list into lists of NTrajectories empty lists, 
    # which can then by passed on to the simulation routine
    if list_of_starting_configurations==[]:
        local_list_of_starting_configurations=[]
        if debug:
            print("No list of starting configurations")
        for n in range(NTrajectories): 
            local_list_of_starting_configurations.append([])
    else:
        local_list_of_starting_configurations = list_of_starting_configurations

    for n in range(NTrajectories):
        if debug:
            print('.', end='', flush=True)
        (t_tr, x_tr, y_tr, vx_tr, vy_tr, 
         uPot_tr, uKin_tr, pPot_tr, pKin_tr, 
         local_list_of_starting_configurations[n]
        ) = Generate_LJ_NVT_MolecularDynamics_Trajectory(d,m,LBox,kT,run_time,
                                                         local_list_of_starting_configurations[n],
                                                         time_step=time_step,
                                                         number_of_time_steps_between_stored_configurations=number_of_time_steps_between_stored_configurations,
                                                         number_of_time_steps_between_velocity_resets=number_of_time_steps_between_velocity_resets)

        # append trajectories to corresponding ensemble lists
        t_tr_ens.append(t_tr)
        x_tr_ens.append(x_tr)
        vx_tr_ens.append(vx_tr)
        y_tr_ens.append(y_tr)
        vy_tr_ens.append(vy_tr)
        uKin_tr_ens.append(uKin_tr)
        uPot_tr_ens.append(uPot_tr)
        pKin_tr_ens.append(pKin_tr)
        pPot_tr_ens.append(pPot_tr)
    
    if debug:
        print("")
    t_tr_ens = np.array(t_tr_ens)
    x_tr_ens = np.array(x_tr_ens)
    y_tr_ens = np.array(y_tr_ens)
    vx_tr_ens = np.array(vx_tr_ens)
    vy_tr_ens = np.array(vy_tr_ens)
    uKin_tr_ens = np.array(uKin_tr_ens)
    uPot_tr_ens = np.array(uPot_tr_ens)
    pKin_tr_ens = np.array(pKin_tr_ens)
    pPot_tr_ens = np.array(pPot_tr_ens)
    
    return (t_tr_ens, x_tr_ens, y_tr_ens, vx_tr_ens, vy_tr_ens, 
            uPot_tr_ens, uKin_tr_ens, pPot_tr_ens, pKin_tr_ens, 
            local_list_of_starting_configurations)

#######Getting compressibilty from pressure fluctuations########

def Compressibility_from_pressure_fluctuations_in_NVT(d, m, NParticles, LBox, kT, pPot, pHyper, pKin=0):
    """
    Returns an estimate of the compressibility based on the analysis of the fluctuations of
    the virial pressure in the canonical ensemble
    """
    if isinstance(pKin, np.ndarray):
        one_over_beta_T = 2 * NParticles * kT / d / LBox ** d - np.var(pPot + pKin) * LBox ** d / kT + np.mean(
            pPot + pKin + pHyper)
    else:
        one_over_beta_T = - np.var(pPot) * LBox ** d / kT + np.mean(pPot + pHyper)

    return 1 / one_over_beta_T


def Compressibility_from_pressure_fluctuations_in_NVT(d, m, NParticles, LBox, kT, pPot, pHyper, pKin=0):
    """
    Returns an estimate of the compressibility based on the analysis of the fluctuations of
    the virial pressure in the canonical ensemble
    """
    if isinstance(pKin, np.ndarray):
        beta_T = 1. / (2 * NParticles * kT / d / LBox ** d - np.var(pPot + pKin, axis=-1) * LBox ** d / kT + np.mean(
            pPot + pKin + pHyper, axis=-1))
    else:
        beta_T = 1. / (- np.var(pPot, axis=-1) * LBox ** d / kT + np.mean(pPot + pHyper, axis=-1))

    if pPot.ndim == 1:
        # data for one trajectories
        return beta_T
    elif pPot.ndim == 2:
        # data for an ensemble of trajectories
        NTrajectories = beta_T.size
        return np.mean(beta_T), np.std(beta_T) / np.sqrt(NTrajectories)



######################################
############## DENSITY ###############
######################################

#######Slow but useful#######

def CellIndices(NCells, x, xBox):
    (xmin, xmax, LBox) = MD.BoxDimensions(xBox)
    return np.floor(NCells * (x - xmin) / LBox).astype(int)


def SortParticlesIntoGrid(NCells, x, y, xBox, yBox, x_pbc=False, y_pbc=False, debug=False):
    cells = [[set() for _ in range(NCells)] for _ in range(NCells)]  # create list of empty sets #
    nSorted = 0

    (xmin, xmax, LBox) = MD.BoxDimensions(xBox)
    if x_pbc:
        random_box_offset = LBox * np.random.random()  # to break Gallilean invariance
        xmin = random_box_offset
        xmax = xmin + LBox
        x_sort = MD.FoldIntoBox((xmin, xmax), x)  # fold into box / minimum image convention for collisions #
    else:
        x_sort = np.copy(x)
    i = CellIndices(NCells, x_sort, (xmin, xmax))

    (ymin, ymax, LBox) = MD.BoxDimensions(yBox)
    if y_pbc:
        random_box_offset = LBox * np.random.random()  # to break Gallilean invariance
        ymin = random_box_offset
        ymax = ymin + LBox
        y_sort = MD.FoldIntoBox((ymin, ymax), y)  # fold into box / minimum image convention for collisions #
    else:
        y_sort = np.copy(y)
    j = CellIndices(NCells, y_sort, (ymin, ymax))

    for n in range(len(x)):
        if i[n] in range(NCells) and j[n] in range(NCells):
            cells[i[n]][j[n]].add(n)  # add particle index n to the set of particles in cell (i,j) #
            nSorted += 1
    if debug:
        print(nSorted, " of ", len(x), " particles sorted into the grid")
    return cells

####### FastV1 #######


def CellOccupancyV1(NParticles,NCells, xx, yy, xBox, yBox, x_pbc=False, y_pbc=False, debug=False):
    '''
        Returns the occupancy of NCells x NCells cells
        xBox = (xmin,xmax) denotes the simulation box
        xBox = LBox corresponds to a box of (0,LBox)
        if x_pbc==True
            particles are sorted according to their folded positions,
            where the box position is randomly shifted to break Gallilein invariance
        else:
            particles inside the box are sorted according to their absolute positions
        the same rules apply in y-direction
    '''

    # get box dimensions
    (xmin, xmax, LBox) = MD.BoxDimensions(xBox)
    (ymin, ymax, LBox) = MD.BoxDimensions(yBox)

    # sort particles into cells
    x = xx.copy()
    if x_pbc:
        x += LBox / NCells * (np.random.random() - 0.5)  # to break Gallilean invariance #
        x -= np.floor(x / LBox) * LBox  # fold into box / minimum image convention for collisions #
    y = yy.copy()
    if y_pbc:
        y += LBox / NCells * (np.random.random() - 0.5)  # to break Gallilean invariance #
        y -= np.floor(y / LBox) * LBox
    # cell numbers corresponding to x- and y-coordinates
    i = np.floor(NCells * x / LBox).astype(int)
    j = np.floor(NCells * y / LBox).astype(int)
    # combine into absolute cell number
    row = i + NCells * j
    if debug:
        print("row = ", row)
    # particle numbers
    col = np.array(range(NParticles))
    # combine into projector from particles to cells
    data = np.ones(NParticles)
    sparse_cell_projector = sparse.coo_matrix((data, (row, col)), shape=(NCells ** 2, NParticles))
    if debug:
        print("projector = ", sparse_cell_projector.todense())

    return sparse_cell_projector.A.sum(axis=1)

def CellOccupancy_vec(NCells, xx, yy, xBox, yBox, x_pbc=False, y_pbc=False, debug=False):
    if xx.ndim == 1:
        nOcc = CellOccupancy(NCells, xx, yy, xBox, yBox, x_pbc, y_pbc, debug)
    elif xx.ndim == 2:
        # trajectory
        nOcc = []
        for i in range(xx.shape[0]):
            nOcc.append(CellOccupancy(NCells, xx[i], yy[i], xBox, yBox, x_pbc, y_pbc, debug))
    elif xx.ndim == 3:
        # trajectory
        nOcc = []
        for i in range(xx.shape[0]):
            nOcc_tr = []
            for j in range(xx.shape[1]):
                nOcc_tr.append(CellOccupancy(NCells, xx[i, j], yy[i, j], xBox, yBox, x_pbc, y_pbc, debug))
            nOcc.append(nOcc_tr)
    else:
        print("CellOccupancy_vec not defined beyond the ensemble level")
    return np.array(nOcc)


######## Final Version #######


def CellOccupancy(NCells, xx, yy, xBox, yBox, x_pbc=False, y_pbc=False, debug=False):
    '''
        Returns the occupancy of NCells x NCells cells
        xBox = (xmin,xmax) denotes the simulation box
        xBox = LBox corresponds to a box of (0,LBox)
        if x_pbc==True
            particles are sorted according to their folded positions,
            where the box position is randomly shifted to break Gallilein invariance
        else:
            particles inside the box are sorted according to their absolute positions
        the same rules apply in y-direction
        The function can be applied to individual conformations, trajectories and ensembles
            and returns correspondingly shaped arrays
    '''

    def individualCellOccupancy(NCells, xx, yy, xBox, yBox, x_pbc=False, y_pbc=False, debug=False):
        # sort particles into cells

        # Keep cells approximately square
        (xmin, xmax, XBox) = MD.BoxDimensions(xBox)
        (ymin, ymax, YBox) = MD.BoxDimensions(yBox)

        LBox = np.sqrt(XBox * YBox)
        NCellsX = np.int(np.round(NCells * XBox / LBox))
        NCellsY = np.int(np.round(NCells * YBox / LBox))

        NParticles = len(xx)
        # sort particles into cells
        x = xx.copy()
        if x_pbc:
            x += XBox / NCellsX * (np.random.random() - 0.5)  # to break Gallilean invariance #
            x -= np.floor(x / XBox) * XBox  # fold into box / minimum image convention for collisions #
        y = yy.copy()
        if y_pbc:
            y += YBox / NCellsY * (np.random.random() - 0.5)  # to break Gallilean invariance #
            y -= np.floor(y / YBox) * YBox
        # cell numbers corresponding to x- and y-coordinates
        i = np.floor(NCellsX * x / XBox).astype(int)
        j = np.floor(NCellsY * y / YBox).astype(int)
        # combine into absolute cell number
        row = i + NCellsX * j
        if debug:
            print("row = ", row)
        # particle numbers
        col = np.array(range(NParticles))
        # combine into projector from particles to cells
        data = np.ones(NParticles)
        sparse_cell_projector = sparse.coo_matrix((data, (row, col)), shape=(NCellsX * NCellsY, NParticles))
        if debug:
            print("projector = ", sparse_cell_projector.todense())

        return sparse_cell_projector.A.sum(axis=1)

        # vectorize application of individualCellOccupancy

    if xx.ndim == 1:
        nOcc = individualCellOccupancy(NCells, xx, yy, xBox, yBox, x_pbc, y_pbc, debug)
    elif xx.ndim == 2:
        # trajectory
        nOcc = []
        for i in range(xx.shape[0]):
            nOcc.append(individualCellOccupancy(NCells, xx[i], yy[i], xBox, yBox, x_pbc, y_pbc, debug))
    elif xx.ndim == 3:
        # trajectory
        nOcc = []
        for i in range(xx.shape[0]):
            nOcc_tr = []
            for j in range(xx.shape[1]):
                nOcc_tr.append(individualCellOccupancy(NCells, xx[i, j], yy[i, j], xBox, yBox, x_pbc, y_pbc, debug))
            nOcc.append(nOcc_tr)
    else:
        print("CellOccupancy_vec not defined beyond the ensemble level")
    return np.array(nOcc)


######################################
############# DIFFUSION ##############
######################################

####### Getting the MSD of a particule #######

def MeanSquareDisplacements(t_tr,x_tr):
    """
    Returns the particle and time average mean-square displacement < (x(t)-x(0))**2 > 
    in one Cartesian direction for a trajectory of x- or y-positions
    """
    
    NParticles = x_tr.shape[-1]
    length_of_x_tr = x_tr.shape[-2]
    length_of_t_tr = t_tr.shape[-1]
    local_x_tr = np.copy(x_tr)

    if x_tr.ndim>2:
        # data for an ensemble of trajectories
        NTrajectories = x_tr.shape[-3]
        local_x_tr = local_x_tr.transpose(1,0,2)
        # so that the time axis is always the first axis (or rather axis=0)
    else:
        NTrajectories = 1
        
    msd = []
    delta_t = []
    
    for n in range(1,length_of_x_tr//2):
        
        n_t = n*length_of_t_tr//length_of_x_tr  # because sometimes configurations are not stored for each time step
        if t_tr.ndim==1:
            # data for one trajectories
            delta_t.append(t_tr[n_t]-t_tr[0])
        elif t_tr.ndim==2:
            # data for an ensemble of trajectories
            delta_t.append(t_tr[0,n_t]-t_tr[0,0])
        delta_x2 = ( local_x_tr - np.roll(local_x_tr,-n,axis=0) )**2
        msd.append(np.mean(delta_x2[:length_of_x_tr-n]))
        
    return np.array(delta_t), np.array(msd)

######################################
##### Pair correlation function ######
######################################

def UnitHyperSphereSurface(d): return 2.*np.pi**(d/2.)/gamma(d/2.)
def UnitHyperSphereVolume(d): return np.pi**(d/2.)/gamma(d/2.+1)


def Radial_distribution_function(d, LBox, pos, r_range=(0.0, 5.0), bins=50, debug=False):
    """
    returns the pair correlation function as a function of the positions of all particles
    """
    r = np.array(pos)

    if r.ndim > d:
        # data for a trajectory or an ensemble of trajectories
        NParticles = r.shape[-1]
        NDataSets = r.size // d // NParticles

    elif r.ndim == d:
        # data for an individual snapshot
        NParticles = r.shape[-1]
        NDataSets = 1

    else:
        print("provided coordinates are dimension ", r.ndim)
        print("expected dimension ", d)
        return

    rho = (NParticles - 1) / LBox ** 2
    # this is the density of all the OTHER particles, which a test particle can see

    if debug:
        print(NParticles, "particles in one configuration")
        print(NDataSets, "data sets")
        print("r_range", r_range)
        print("bins", bins)

    hist = np.zeros(bins)

    for k in range(1, NParticles):
        delta_r_pair_vectors = MD.MinimumImage(LBox, r - np.roll(r, k, axis=-1))
        delta_r_pair_sqr = delta_r_pair_vectors ** 2  # array of Cartesian components of squared distances
        if d > 1:
            delta_r_pair_sqr = np.sum(delta_r_pair_sqr, axis=0)  # add up Cartesian components in d>1
        delta_r_pair = np.sqrt(delta_r_pair_sqr)
        hist_k, bin_edges = np.histogram(delta_r_pair, bins, r_range, normed=None, weights=None, density=None)
        hist += hist_k

    if debug:
        print(np.sum(hist), " pair distances")
        print(NParticles * (NParticles - 1) * NDataSets, " expected")

    # expected number of pair distances in spherical shell
    normalization = (
                UnitHyperSphereSurface(d) * MD.BinCenters(bin_edges) ** (d - 1) * (r_range[1] - r_range[0]) / bins *
                # bin volume
                rho *
                # density
                NDataSets * NParticles
                # number of test particles = number of datasets * number of particles
                )

    if debug:
        print(np.sum(normalization), "expected pair distances in an ideal gas")

    return hist / normalization, bin_edges

####### NNN ########

def Number_of_nearest_neighbors(d, LBox, pos, r_max, debug=False):
    """
    returns for each particle and each snapshot the number of other particles found within a radius of r_max
    """
    r = np.array(pos)

    if r.ndim > d:
        # data for a trajectory or an ensemble of trajectories
        NParticles = r.shape[-1]
        NDataSets = r.size // d // NParticles

    elif r.ndim == d:
        # data for an individual snapshot
        NParticles = r.shape[-1]
        NDataSets = 1

    else:
        print("provided coordinates are dimension ", r.ndim)
        print("expected dimension ", d)
        return

    if d == 1:
        # output has the same format as 1d position information:
        # instead of a float indicating a particle position we now have an integer indicating the number of close-by neighbors
        NNN = 0 * np.copy(r).astype(int)
    else:
        # output has the same format as 1d position information:
        # instead of a float indicating a particle position we now have an integer indicating the number of close-by neighbors
        NNN = 0 * np.copy(r[0]).astype(int)

    if debug:
        print(NParticles, "particles in one configuration")
        print(NDataSets, "data sets")
        print("r_max", r_max)

    for k in range(1, NParticles):
        delta_r_pair_vectors = MD.MinimumImage(LBox, r - np.roll(r, k, axis=-1))
        delta_r_pair_sqr = delta_r_pair_vectors ** 2  # array of Cartesian components of squared distances
        if d > 1:
            delta_r_pair_sqr = np.sum(delta_r_pair_sqr, axis=0)  # add up Cartesian components in d>1

        pair_in_range = np.round((np.sign(r_max ** 2 - delta_r_pair_sqr) + 1.0) / 2.0).astype(int)

        NNN += pair_in_range

    return NNN

######################################
###### LJ-Pot useful functions #######
######################################

####### Energy #######

def U_LJ(d, epsilon, sigma, distance_vector):
    """
    Lennard-Jones potential energy in d dimensions
    d is the embedding dimension. Needed to distinguish the case of 2 1d distance vectors from 1 2d distance vector.
    epsilon is the energy scale of the LJ potential and can be
        a scalar for a single interaction or if all interactions have the same epsilon
        an array of the length of the array of distance vectors
    sigma is the interaction range and can be
        a scalar for a single interaction or if all interactions have the same epsilon
        an array of the length of the array of distance vectors
    distance_vector are the instanteneous distances
        in d = 1 distance_vector = delta_x
        in d>1 distance_vector has to be of the form distance_vector = (delta_x,delta_x) or distance_vector = [delta_x,delta_x]
        where delta_x and delta_y can be
            a scalar for a single interaction
            an array for several interactions to be evaluated simultaneously
    The function returns 4*epsilon*((sigma/r)**(-12)-(sigma/r)**-6) where u has the same format as delta_x
    """
    eps = np.array(epsilon)
    sig = np.array(sigma)
    delta_r = np.array(distance_vector)  # array of scalar(d=1) or vector (d>1) distances
    delta_r_sqr = delta_r ** 2  # array of Cartesian components of squared distances
    if d > 1:
        delta_r_sqr = np.sum(delta_r_sqr, axis=0)  # add up Cartesian components in d>1
    relative_inverse_squared_distance = sigma ** 2 / delta_r_sqr
    u = 4 * epsilon * (relative_inverse_squared_distance ** 6 - relative_inverse_squared_distance ** 3)
    return u

####### Force #######

def f_LJ(d, epsilon, sigma, distance_vector, debug=False):
    """
    Lennard-Jones force in d dimensions
    d is the embedding dimension. Needed to distinguish the case of 2 1d distance vectors from 1 2d distance vector.
    epsilon is the energy scale of the LJ potential and can be
        a scalar for a single interaction or if all interactions have the same epsilon
        an array of the length of the array of distance vectors
    sigma is the interaction range and can be
        a scalar for a single interaction or if all interactions have the same epsilon
        an array of the length of the array of distance vectors
    distance_vector are the instanteneous distances
        in d = 1 distance_vector = delta_x
        in d>1 distance_vector has to be of the form distance_vector = (delta_x,delta_x) or distance_vector = [delta_x,delta_x]
        where delta_x and delta_y can be
            a scalar for a single interaction
            an array for several interactions to be evaluated simultaneously
    The function returns -24*epsilon*(2*(sigma/r)**(-12)-(sigma/r)**-6) * distance_vector/r**2
        where f has the same format as distance_vector
        and r = |distance_vector|
    """
    eps = np.array(epsilon)
    sig = np.array(sigma)
    delta_r = np.array(distance_vector)  # array of scalar(d=1) or vector (d>1) distances
    if debug:
        print(delta_r)
    delta_r_sqr = delta_r ** 2  # array of Cartesian components of squared distances
    if d > 1:
        delta_r_sqr = np.sum(delta_r_sqr, axis=0)  # add up Cartesian components in d>1
    relative_inverse_squared_distance = sigma ** 2 / delta_r_sqr
    f = 24 * epsilon * (
                2 * relative_inverse_squared_distance ** 6 - relative_inverse_squared_distance ** 3) / delta_r_sqr * delta_r
    return f

####### Initial state = Grid (metastable) #######

def GridPositionsIn2d(xBox, yBox, NParticles, debug=False):
    """
    Returns two arrays of x- and y-positions for NParticles in the interval
        xBox = [xmin,xmax) or xBox * [0,1), if xBox is a float or integer
        yBox = [xmin,xmax) or xBox * [0,1), if xBox is a float or integer
    """
    n = np.ceil(np.sqrt(NParticles))  # number of particles in a row or column of our square grid
    (xmin, xmax, XBox) = MD.BoxDimensions(xBox)
    ax = XBox / n
    (ymin, ymax, YBox) = MD.BoxDimensions(yBox)
    ay = XBox / n

    if debug:
        print("Lattice constant: ", ax, ay)

    x = MD.FoldIntoBox(xBox, xmin + ax * (np.arange(NParticles) + 0.5))
    y = ymin + ay * (np.arange(NParticles) // n + 0.5)

    return x, y

####### Loops (numpy versions) #######


def LJ_forces_as_a_function_of_positions(d, epsilon, sigma, LBox, r, debug=False):
    """
    returns the LJ force acting on each particle as a function of the positions of all particles
    """
    r = np.array(r)
    N = r.shape[-1]
    f = 0 * np.copy(r)  # initialise force array with the same shape as position array

    if debug:
        print(N)

    for k in range(1, N):
        delta_r_pair = MD.MinimumImage(LBox, r - np.roll(r, k, axis=-1))
        fpair = f_LJ(d, epsilon, sigma, delta_r_pair)
        f += fpair
        f -= np.roll(fpair, -k, axis=-1)
    return f / 2


def LJ_energies_as_a_function_of_positions(d, NParticles, epsilon, sigma, LBox, r, debug=False):
    """
    returns the LJ interaction energy for each particle as a function of the positions of all particles
    """
    r = np.array(r)
    if d == 1:
        u = 0 * np.copy(r)  # initialise energy array with the same shape as the d=1 position array
    else:
        u = 0 * np.copy(r[0])  # initialise energy array with the same shape as the first Cartesian component
        # of the position array
    N = r.shape[-1]

    if (debug):
        print(N, NParticles, epsilon, sigma)

    for k in range(1, N):
        delta_r_pair = MD.MinimumImage(LBox, r - np.roll(r, k, axis=-1))
        upair = U_LJ(d, epsilon, sigma, delta_r_pair)
        u += upair / 2
        u += np.roll(upair, -k, axis=-1) / 2
    return u / 2


def LJ_energy_as_a_function_of_positions(d, epsilon, sigma, LBox, r):
    """
    returns the total LJ interaction energy as a function of the positions of all particles
    """
    return np.sum(LJ_energies_as_a_function_of_positions(d, epsilon, sigma, LBox, r), axis=-1)