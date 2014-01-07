import sys
import numpy as np
import math as ma
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from matplotlib.patches import Ellipse
import emcee as em
from emcee.utils import MPIPool
import os
import quantities as pq
import collections
from scipy import optimize
from PyAstronomy import pyasl
import argparse

parser = argparse.ArgumentParser(description="Fit G2's orbit.")
parser.add_argument('--samerp',           dest='samerp',    help='G2 and Candidate forced to have same rp', 					default=False, action='store_true')
parser.add_argument('--samew',            dest='samew',     help='G2 and Candidate forced to have same w',  					default=False, action='store_true')
parser.add_argument('--noproprec',        dest='noproprec', help='Candidate not allowed to precess prograde relative to G2',  default=False, action='store_true')
parser.add_argument('--prior',            dest='prior',     help='Use prior for Sgr A* properties',                     default=False, action='store_true')
parser.add_argument('--dataset',          dest='dataset',   help='Dataset name',                                        default='',    type=str)
parser.add_argument('--id',               dest='id',        help='ID of star in dataset',  				                default=-1,    type=int)
parser.add_argument('--nwalkers',         dest='nwalkers',  help='Number of walkers',  				                    default=-1,    type=int)
parser.add_argument('--nsteps',           dest='nsteps',    help='Number of steps',  				                    default=-1,    type=int)
parser.add_argument('--inputs',           dest='inputs',    help='Which data to use for S2/G2',  			            default=-1,    type=int)
args = parser.parse_args()

def neg_obj_func(x, ptimes, vtimes):
	return -obj_func(x, ptimes, vtimes)

def perturb(x, m):
	px = x + m*np.abs(x)*np.random.randn(x.shape[0])
	return px

def flatten(l):
    for el in l:
        if isinstance(el, collections.Iterable) and not isinstance(el, basestring):
            for sub in flatten(el):
                yield sub
        else:
            yield el

def get_star_ke(elements, lmh):
	ret = [[] for _ in xrange(elements.shape[0])]

	for i, e in enumerate(elements):
		a = e[0]
		mh = 10.**lmh
		loce = 1. - 10.**e[1]
		period = 2.*np.pi*np.sqrt(a**3/(G*mh))
		loctau = e[3]*period
		ret[i] = pyasl.KeplerEllipse(a, period, e=loce, Omega=e[2], tau=loctau, w=e[4], i=e[5])

	return ret

def chunks(l, n):
	return np.array([l[i:i+n] for i in range(0, len(l), n)])

def obj_func(x, times, types, measurements, errors, objects, coords, varia, prior, mode, save):
	global vecs, vecs2, temp, elements, variances, mhx, mhy, mhz, mhvx, mhvy, mhvz, lmh
	lmh =  x[0]
	mhz =  x[1]
	variances = x[2:nvaria+2]
	if min(variances) < 0: return float("-inf")
	mhx =  x[nvaria+2:5*ncoords+nvaria+2:5]
	mhy =  x[nvaria+3:5*ncoords+nvaria+2:5]
	mhvx = x[nvaria+4:5*ncoords+nvaria+2:5]
	mhvy = x[nvaria+5:5*ncoords+nvaria+2:5]
	mhvz = x[nvaria+6:5*ncoords+nvaria+2:5]
	elements = chunks(x[5*ncoords+nvaria+2:], 6)

	#print lmh
	#print mhz
	#print variances
	#print mhx
	#print mhy
	#print mhvx
	#print mhvy
	#print mhvz
	#print elements
	#sys.exit(0)

	mh = 10.**lmh

	if nobjects >= 3:
		# Hacky, third object has restricted parameters depending on object 2.
		if args.samerp:
			if args.samew:
				elements[2] = [elements[2][0],0.,0.,0.,0.,0.]
			else:
				elements[2] = [elements[2][0],0.,0.,0.,elements[2][1],0.]
		else:
			if args.samew:
				elements[2] = [elements[2][0],elements[2][1],0.,0.,0.,0.]
			else:
				# For floating w
				elements[2] = [elements[2][0],elements[2][1],0.,0.,elements[2][2],0.]

	for i, e in enumerate(elements):
		if e[0] < 1.e13: return float("-inf")
		if e[1] > 0.: return float("-inf")
		if e[2] > 360. or e[2] < 0.: return float("-inf")
		if e[3] < 0. or e[3] > 1.: return float("-inf")
		if e[4] > 360. or e[4] < 0.: return float("-inf")
		if e[5] > 360. or e[5] < 0.: return float("-inf")

	if nobjects >= 3:
		a1 = elements[1][0]
		period1 = 2.*np.pi*np.sqrt(a1**3/(G*mh))
		a2 = elements[2][0]
		period2 = 2.*np.pi*np.sqrt(a2**3/(G*mh))
		if a1 > a2: return float("-inf")
		if args.samerp:
			# G2 and Candidate have same Rp.
			elements[2][1] = elements[1][1] + np.log10(a1/a2)
		else:
			# Allow G2 and Candidate to have different Rp, but Rp must be less than a critical value (tidal radius).
			rp = a2*10.**elements[2][1]
			if elements[2][1] < elements[1][1] or rp > rpmax: return float("-inf")
		if elements[2][1] > 0.: return float("-inf")
		elements[2][2] = elements[1][2]
		#elements[2][3] = elements[1][3]*period1/period2
		elements[2][3] = np.mod(g2reftime - np.mod(g2reftime - elements[1][3]*period1, period1), period2)/period2
		#elements[2][3] = (1. - (1.-elements[1][3])*period1/period2)
		if args.samew:
			elements[2][4] = elements[1][4]
		elif args.noproprec:
			# Retrograde precession only, GR won't be important
			if elements[2][4] > elements[1][4]: return float("-inf")
		elements[2][5] = elements[1][5]

	# Skip obj func calculation in this mode
	if mode == 2:
		return float("-inf")

	kes = get_star_ke(elements, lmh)

	val = 0.
	vecs = [[] for _ in xrange(times.shape[0])]

	#vecs2 = [[] for _ in xrange(times.shape[0])]
	for i, stimes in enumerate(times):
		ci = coords[i]
		gi = objects[i]
		vi = varia[i]
		if types[i] == 'pxy':
			posvec = kes[gi].xyzPos(stimes)
			dat = posvec[:,:2]

			dat[:,0] += 2.*mhz*np.tan(mhx[ci]*iasec) + mhvx[ci]*km*stimes
			dat[:,1] += 2.*mhz*np.tan(mhy[ci]*iasec) + mhvy[ci]*km*stimes

			if save: vecs[i] = np.copy(dat)

			#if objects[i] == 1:
			#	velvec = kes[gi].xyzVel(stimes)
			#	dists = np.array([np.linalg.norm(x) for x in posvec])
			#	velvec = np.array([np.array(x/np.linalg.norm(x)) for x in velvec])
			#	nvaria = 0.3*velvec*np.transpose(np.tile(dists,(3,1)))
			#	dat[:,0] += nvaria[:,0]
			#	dat[:,1] += nvaria[:,1]
			#	vecs2[i] = np.copy(dat)

			#val += np.sum((((dat[:,0] - measurements[i][:,0]*mhz)/(errors[i][:,0]*mhz)))**2)
			#val += np.sum((((dat[:,1] - measurements[i][:,1]*mhz)/(errors[i][:,1]*mhz)))**2)
			# For max likelihood

			if vi != -1:
				variancepxy2 = variances[vi]**2
			else:
				variancepxy2 = 0.

			val += np.sum((dat[:,0]/mhz - measurements[i][:,0])**2/(errors[i][:,0]**2 + variancepxy2))
			val += np.sum((dat[:,1]/mhz - measurements[i][:,1])**2/(errors[i][:,1]**2 + variancepxy2))
			if mode == 1:
				val += 0.5*np.sum(np.log(errors[i][:,0]**2 + variancepxy2))
				val += 0.5*np.sum(np.log(errors[i][:,1]**2 + variancepxy2))
		elif types[i] == 'vz':
			dat = kes[gi].xyzVel(stimes)[:,2:3]
			dat += mhvz[ci]*km
			if save: vecs[i] = np.copy(dat)

			if vi != -1:
				variancevz2 = (variances[vi]*km)**2
			else:
				variancevz2 = 0.

			val += np.sum((dat - measurements[i])**2/(errors[i]**2 + variancevz2))
			if mode == 1:
				val += 0.5*np.sum(np.log(errors[i]**2 + variancevz2))
		elif types[i] == 'vxy':
			dat = kes[gi].xyzVel(stimes)[:,0:2]
			dat[:,0] += mhvx[ci]*km
			dat[:,1] += mhvy[ci]*km
			if save: vecs[i] = np.copy(dat)
			#print 'vxy', np.sum(((dat[:,0] - measurements[i][:,0]*mhz)/(errors[i][:,0]*mhz))**2), np.sum(((dat[:,1] - measurements[i][:,1]*mhz)/(errors[i][:,1]*mhz))**2)
			val += np.sum(((dat[:,0] - measurements[i][:,0]*mhz)/(errors[i][:,0]*mhz))**2)
			val += np.sum(((dat[:,1] - measurements[i][:,1]*mhz)/(errors[i][:,1]*mhz))**2)
		else:
			print 'Error, invalid type'
			sys.exit(0)

	if prior:
		pval = -0.5*((mh - 4.31e6*msun)/(0.42e6*msun))**2 - 0.5*((mhz - 8.33*kpc)/(0.35*kpc))**2
		for ci in xrange(ncoords):
			pval = pval - 0.5*(mhvz[ci]/5.)**2
	else:
		pval = 0.

	return -0.5*val + pval#/temp

global temp, elements, vecs, vecs2, variances

# Some options to toggle
rpmax = 1.e300

if args.inputs == -1:
	print 'Error, must specify inputs'
	sys.exit(0)

# User adjustable parameters
if args.nwalkers == -1:
	nwalkers = 64
else:
	nwalkers = args.nwalkers
if args.nsteps == -1:
	nsteps = 1000
else:
	nsteps = args.nsteps
nburn = nsteps/2
t0 = 1.e4

# Constants and units
pq.set_default_units('cgs')

G = pq.constants.G.simplified.magnitude
pc = pq.pc.simplified.magnitude
au = pq.au.simplified.magnitude
yr = pq.year.simplified.magnitude
kpc = 1.e3*pc
mpc = 1.e-3*pc
impc = 1./mpc
km = 1.e5
ikm = 1./km
msun = 1.9891e33
asec = 2.*3600.*180./np.pi
iasec = 1./asec

temp = t0

zscale = 2.
zstretch = ((zscale - 1.) + 1.)**2/zscale
	
nvaria = 2

#cd = os.getcwd()
#cd = '/pfs/james/g2fit'
cd = os.path.dirname(os.path.realpath(__file__))+"/observations/"

# Load data files
s2data = np.loadtxt(cd+"phifer-2013/S2.points")
s2data2 = np.loadtxt(cd+"gillessen-2009b/S2.points")
s2vdata = np.loadtxt(cd+"phifer-2013/S2.rv")
s2vdata2 = np.loadtxt(cd+"gillessen-2009b/S2.rv")
g2data = np.loadtxt(cd+"phifer-2013/G2.points")
g2data2 = np.loadtxt(cd+"gillessen-2013/G2.points.brg")
g2vdata = np.loadtxt(cd+"phifer-2013/G2.rv")
g2vdata2 = np.loadtxt(cd+"gillessen-2013/G2.rv")

if args.dataset == 'sch':
	if args.id == -1:
		args.id = 20
	else:
		canddata = np.loadtxt(cd+"schoedel-2009/Sch"+str(args.id)+".points")
		candvxydata = np.loadtxt(cd+"schoedel-2009/Sch"+str(args.id)+".vxy")
elif args.dataset == 'yel':
	if args.id == -1:
		args.id = 20
	else:
		canddata = np.loadtxt(cd+"yelda-2010/Yelda"+str(args.id)+".points")
		candvxydata = np.loadtxt(cd+"yelda-2010/Yelda"+str(args.id)+".vxy")
elif args.dataset == 'lu':
	if args.id == -1:
		args.id = 1
	else:
		canddata = np.loadtxt(cd+"lu-2009/Lu"+str(args.id)+".points")
		candvxydata = np.loadtxt(cd+"lu-2009/Lu"+str(args.id)+".vxy")
		candvdata = np.loadtxt(cd+"lu-2009/Lu"+str(args.id)+".rv")
else:
	print "Invalid dataset selected"
	sys.exit(0)

# Convert data units
s2data[:,0] = s2data[:,0]*yr
s2data[:,1] = -s2data[:,1] #Sign needs to be flipped according to Leo's e-mail
s2data[:,1:] = 2.*np.tan(s2data[:,1:]*iasec)

s2data2[:,0] = s2data2[:,0]*yr
s2data2[:,1:] = 0.001*s2data2[:,1:] #Sign does NOT need to be flipped
s2data2[:,1:] = 2.*np.tan(s2data2[:,1:]*iasec)

s2vdata[:,0] = s2vdata[:,0]*yr
s2vdata[:,1] = -s2vdata[:,1] #Positive radial velocity indicates recession (negative v_z)
s2vdata[:,1:] = s2vdata[:,1:]*km

s2vdata2[:,0] = s2vdata2[:,0]*yr
s2vdata2[:,1] = -s2vdata2[:,1] #Positive radial velocity indicates recession (negative v_z)
s2vdata2[:,1:] = s2vdata2[:,1:]*km

g2data[:,0] = g2data[:,0]*yr
g2data[:,1] = -g2data[:,1] #Sign needs to be flipped according to Leo's e-mail
#g2data[:,2] = g2data[:,2] + 0.01 #Just testing some extra error in G2's position
g2data[:,1:] = 2.*np.tan(g2data[:,1:]*iasec)

# Adding some positional uncertainty to G2
#g2data[:,3:5] = g2data[:,3:5] + 200.*au/(8.3*kpc)

g2data2[:,0] = g2data2[:,0]*yr
g2data2[:,1] = -g2data2[:,1] #Sign needs to be flipped according to Leo's e-mail
g2data2[:,1:] = 2.*np.tan(g2data2[:,1:]*iasec)

# Adding some positional uncertainty to G2
#g2data2[:,3:5] = g2data2[:,3:5] + 200.*au/(8.3*kpc)

g2vdata[:,0] = g2vdata[:,0]*yr
g2vdata[:,1] = -g2vdata[:,1] #Positive radial velocity indicates recession (negative v_z)
g2vdata[:,1:] = g2vdata[:,1:]*km 

# Adding some velocity uncertainty to G2
#g2vdata[:,2] = g2vdata[:,2] + 100*km

g2vdata2[:,0] = g2vdata2[:,0]*yr
g2vdata2[:,1] = -g2vdata2[:,1] #Positive radial velocity indicates recession (negative v_z)
g2vdata2[:,1:] = g2vdata2[:,1:]*km 

# Adding some velocity uncertainty to G2
#g2vdata2[:,2] = g2vdata2[:,2] + 100*km

# From Schodel 2009
#canddata = np.reshape(canddata, (-1, 5))
#canddata[:,0] = canddata[:,0]*yr
#canddata[:,1:] = 2.*np.tan(canddata[:,1:]*iasec)
#candvxydata = np.reshape(candvxydata, (-1, 5))
#candvxydata[:,0] = candvxydata[:,0]*yr
#candvxydata[:,1:] = candvxydata[:,1:]*km
## From Schodel 2009, 8 kpc was assumed, divide by their distance and multiply back measured distance
#candvxydata[:,1:] = candvxydata[:,1:]/(8.*kpc)

# From Gillessen 2009
canddata = np.reshape(canddata, (-1, 5))
canddata[:,0] = canddata[:,0]*yr
canddata[:,1:] = 2.*np.tan(canddata[:,1:]*iasec)
candvxydata = np.reshape(candvxydata, (-1, 5))
candvxydata[:,0] = candvxydata[:,0]*yr
candvxydata[:,1:] = 2.*np.tan(candvxydata[:,1:]*iasec)/yr
if args.dataset == 'lu':
	candvdata = np.reshape(candvdata, (-1, 3))
	candvdata[:,0] = candvdata[:,0]*yr
	candvdata[:,1] = -candvdata[:,1]
	candvdata[:,1:] = candvdata[:,1:]*km

# Both datasets

if args.inputs == 0:
	if args.dataset == 'lu':
		times = np.array([s2data[:,0],s2data2[:,0],g2data[:,0],g2data2[:,0],canddata[:,0],s2vdata[:,0],s2vdata2[:,0],g2vdata[:,0],g2vdata2[:,0],candvxydata[:,0],candvdata[:,0]])
		types = ['pxy','pxy','pxy','pxy','pxy','vz','vz','vz','vz','vxy','vz']
		measurements = [s2data[:,1:3],s2data2[:,1:3],g2data[:,1:3],g2data2[:,1:3],canddata[:,1:3],s2vdata[:,1:2],s2vdata2[:,1:2],g2vdata[:,1:2],g2vdata2[:,1:2],candvxydata[:,1:3],candvdata[:,1:2]]
		errors = [s2data[:,3:5],s2data2[:,3:5],g2data[:,3:5],g2data2[:,3:5],canddata[:,3:5],s2vdata[:,2:3],s2vdata2[:,2:3],g2vdata[:,2:3],g2vdata2[:,2:3],candvxydata[:,3:5],candvdata[:,2:3]]
		objects = [0,0,1,1,2,0,0,1,1,2,2]
		coords = [0,1,0,1,1,0,1,0,1,1,0]
		kind = [0,0,1]
		names = ['S2','G2','Candidate']
		varia = [0,1,2,3,-1,-1,-1,4,5,-1,-1]
	else:
		times = np.array([s2data[:,0],s2data2[:,0],g2data[:,0],g2data2[:,0],canddata[:,0],s2vdata[:,0],s2vdata2[:,0],g2vdata[:,0],g2vdata2[:,0],candvxydata[:,0]])
		types = ['pxy','pxy','pxy','pxy','pxy','vz','vz','vz','vz','vxy']
		measurements = [s2data[:,1:3],s2data2[:,1:3],g2data[:,1:3],g2data2[:,1:3],canddata[:,1:3],s2vdata[:,1:2],s2vdata2[:,1:2],g2vdata[:,1:2],g2vdata2[:,1:2],candvxydata[:,1:3]]
		errors = [s2data[:,3:5],s2data2[:,3:5],g2data[:,3:5],g2data2[:,3:5],canddata[:,3:5],s2vdata[:,2:3],s2vdata2[:,2:3],g2vdata[:,2:3],g2vdata2[:,2:3],candvxydata[:,3:5]]
		objects = [0,0,1,1,2,0,0,1,1,2]
		if args.dataset == 'sch' or args.dataset == 'yel':
			coords = [0,1,0,1,1,0,1,0,1,1]
		else:
			coords = [0,1,0,1,0,0,1,0,1,0]
		kind = [0,0,1]
		names = ['S2','G2','Candidate']
		varia = [0,1,2,3,-1,-1,-1,4,5,-1]
# Both datasets minus Candidate
#times = np.array([s2data[:,0],s2data2[:,0],g2data[:,0],g2data2[:,0],s2vdata[:,0],s2vdata2[:,0],g2vdata[:,0],g2vdata2[:,0]])
#types = ['pxy','pxy','pxy','pxy','vz','vz','vz','vz']
#measurements = [s2data[:,1:3],s2data2[:,1:3],g2data[:,1:3],g2data2[:,1:3],s2vdata[:,1:2],s2vdata2[:,1:2],g2vdata[:,1:2],g2vdata2[:,1:2]]
#errors = [s2data[:,3:5],s2data2[:,3:5],g2data[:,3:5],g2data2[:,3:5],s2vdata[:,2:3],s2vdata2[:,2:3],g2vdata[:,2:3],g2vdata2[:,2:3]]
#objects = [0,0,1,1,0,0,1,1]
#coords = [0,1,0,1,0,1,0,1]
#kind = [0,0]
#names = ['S2','G2']
#varia = [0,1,2,3,-1,-1,4,5]
# Both datasets, minus Gillessen G2 position
#times = np.array([s2data[:,0],s2data2[:,0],g2data[:,0],canddata[:,0],s2vdata[:,0],s2vdata2[:,0],g2vdata[:,0],g2vdata2[:,0],candvxydata[:,0]])
#types = ['pxy','pxy','pxy','pxy','vz','vz','vz','vz','vxy']
#measurements = [s2data[:,1:3],s2data2[:,1:3],g2data[:,1:3],canddata[:,1:3],s2vdata[:,1:2],s2vdata2[:,1:2],g2vdata[:,1:2],g2vdata2[:,1:2],candvxydata[:,1:3]]
#errors = [s2data[:,3:5],s2data2[:,3:5],g2data[:,3:5],canddata[:,3:5],s2vdata[:,2:3],s2vdata2[:,2:3],g2vdata[:,2:3],g2vdata2[:,2:3],candvxydata[:,3:5]]
#objects = [0,0,1,2,0,0,1,1,2]
#coords = [0,1,0,1,0,1,0,1,1]
#kind = [0,0,1]
#names = ['S2','G2','Candidate']
#varia = [0,1,2,-1,-1,-1,3,4,-1]
# Just Ghez S2 + G2 + Candidate
elif args.inputs == 1:
	times = np.array([s2data[:,0],g2data[:,0],canddata[:,0],s2vdata[:,0],g2vdata[:,0],candvxydata[:,0]])
	types = ['pxy','pxy','pxy','vz','vz','vxy']
	measurements = [s2data[:,1:3],g2data[:,1:3],canddata[:,1:3],s2vdata[:,1:2],g2vdata[:,1:2],candvxydata[:,1:3]]
	errors = [s2data[:,3:5],g2data[:,3:5],canddata[:,3:5],s2vdata[:,2:3],g2vdata[:,2:3],candvxydata[:,3:5]]
	objects = [0,1,2,0,1,2]
	coords = [0,0,0,0,0,0]
	kind = [0,0,1]
	names = ['S2','G2','Candidate']
	varia = [0,1,-1,2,3,-1]
# Just Genzel S2 + G2 + Candidate
elif args.inputs == 2:
	times = np.array([s2data2[:,0],g2data2[:,0],canddata[:,0],s2vdata2[:,0],g2vdata2[:,0],candvxydata[:,0]])
	types = ['pxy','pxy','pxy','vz','vz','vxy']
	measurements = [s2data2[:,1:3],g2data2[:,1:3],canddata[:,1:3],s2vdata2[:,1:2],g2vdata2[:,1:2],candvxydata[:,1:3]]
	errors = [s2data2[:,3:5],g2data2[:,3:5],canddata[:,3:5],s2vdata2[:,2:3],g2vdata2[:,2:3],candvxydata[:,3:5]]
	objects = [0,1,2,0,1,2]
	coords = [0,0,0,0,0,0]
	kind = [0,0,1]
	names = ['S2','G2','Candidate']
	varia = [0,1,-1,2,3,-1]

zerot = min(list(flatten(times)))
datalen = len(list(flatten(times)))
for i in xrange(len(types)):
	times[i] = times[i] - zerot

g2reftime = min(list(flatten(np.array([g2data[:,0],g2vdata[:,0],g2data2[:,0],g2vdata2[:,0]])))) - zerot

nobjects = len(set(objects))
ncoords = len(set(coords))
nvaria = len(set(varia)) - 1 #Need -1 because some have no variance

#x0 = [np.array([40.,0.01,0.01,200.,8.36*kpc,0.04*pc,0.9999,360.,1.,360.,360.,0.04*pc,0.9999,360.,1.,360.,360.]) * np.random.rand(ndim) for i in xrange(nwalkers)]

#x0 = em.utils.sample_ball([3.98984624e+01, 5.75930617e-03, -1.29642155e-02, -7.20738328e+00, 2.09758430e+01, 6.92943815e+00,
#						   0.001, 0.001, 0., 0., 0.,
#						   2.31820365e+22,
#						   1.51184008e+16, -1.00068439e+00, 2.22895592e+02, 4.29469497e-01, 6.70875379e+01, 3.12620663e+02,
#						   1.05126070e+17, -2.06083486e+00, 2.10324682e+02, 6.39904114e-02, 2.76476139e+02, 6.24846279e+01],
#						  [0.05,0.001,0.001,10.,10.,10.,0.001,0.001,10.,10.,10.,1*kpc,0.0005*pc,0.1,5.,0.01,5.,5.,0.1*pc,0.5,360.,0.1,360.,360.], size=nwalkers)
#x0 = em.utils.sample_ball([3.99298621e+01, 4.50478826e-03, -1.28610303e-02, -5.16967075e+00, 2.64503308e+01, 2.78545427e+00, 2.41850038e+22,
#						   1.53696988e+16, -9.86146230e-01, 2.21992919e+02, 4.34253060e-01, 6.70356104e+01, 3.14016468e+02,
#						   0.04*pc,-1,360.,1.,360.,360.],
#				  		  [0.1,0.0001,0.0001,20.,20.,20.,0.1*kpc,0.0005*pc,0.1,36.,0.1,36.,36.,0.004*pc,0.1,360.,0.1,360.,360.], size=nwalkers)

# One object only
#x0 = em.utils.sample_ball([39.9423209093, 1e+15, 0.00209107565329, 0.00217098237557, 0.837772793736, -1.45335115374, -1.57080647175, 2.61755477547e+22,
#						   1.55338087572e+16, -0.911020170487, 44.2334977541, 0.633314082346, 243.911799883, 43.9361641325
#						   ],
#				  		  [0.1,0.e15,0.0001,0.0001,20.,20.,20.,0.1*kpc,
#						   0.0005*pc,0.1,36.,0.1,36.,36.
#						   ], size=nwalkers)

# Two objects with initial decent guess
#x0 = em.utils.sample_ball([3.99298621e+01, 4.50478826e-03, -1.28610303e-02, -5.16967075e+00, 2.64503308e+01, 2.78545427e+00, 2.41850038e+22,
#						   1.51184008e+16, -1.00068439e+00, 2.22895592e+02, 4.29469497e-01, 6.70875379e+01, 3.12620663e+02,
#						   1.05126070e+17, -2.06083486e+00, 2.10324682e+02, 6.39904114e-02, 2.76476139e+02, 6.24846279e+01],
#				  		  [0.1,0.0001,0.0001,20.,20.,20.,0.1*kpc,
#						   0.0005*pc,0.1,3.6,0.1,3.6,3.6,
#						   0.0004*pc,0.1,3.6,0.1,3.6,3.6], size=nwalkers)

# Two objects with initial decent guess and max. likelihood
#x0 = em.utils.sample_ball([3.99298621e+01, 4.50478826e-03, -1.28610303e-02, -5.16967075e+00, 2.64503308e+01, 2.78545427e+00, 2.41850038e+22, 
#						   1.51184008e+16, -1.00068439e+00, 2.22895592e+02, 4.29469497e-01, 6.70875379e+01, 3.12620663e+02,
#						   1.05126070e+17, -2.06083486e+00, 2.10324682e+02, 6.39904114e-02, 2.76476139e+02, 6.24846279e+01],
#				  		  [0.1,0.0001,0.0001,20.,20.,20.,0.1*kpc,
#						   0.0005*pc,0.1,3.6,0.1,3.6,3.6,
#						   0.0004*pc,0.1,3.6,0.1,3.6,3.6], size=nwalkers)

# Two objects with initial decent guess, includes variable variance
#x0 = em.utils.sample_ball([39.9423209093, 1.e+15, 0.00209107565329, 0.00217098237557, 0.837772793736, -1.45335115374, -1.57080647175, 2.61755477547e+22,
#						   1.55338087572e+16, -0.911020170487, 44.2334977541, 0.633314082346, 243.911799883, 43.9361641325,
#						   1.05126070e+17, -2.06083486e+00, 2.10324682e+02, 6.39904114e-02, 2.76476139e+02, 297.5153721
#						   ],
#				  		  [0.1,1.e15,0.0001,0.0001,20.,20.,20.,0.1*kpc,
#						   0.0005*pc,0.1,36.,0.1,36.,36.,
#						   0.0004*pc,0.1,360.,0.1,360.,360.
#						   ], size=nwalkers)

# Three objects with two fixed
#x0 = em.utils.sample_ball([3.98813163e+01,  2.72299631e-03, -1.17165691e-03, -3.57672290e+00,
#                           -2.22465956e+00,   5.14236228e+00,   2.38341193e+22,   1.48288930e+16,
#                           -9.49282749e-01,   2.22253841e+02,   6.32657523e-01,   6.57978526e+01,
#                           3.14124382e+02,   1.02147631e+17,  -1.59176126e+00,   1.81207572e+02,
#                           7.44247185e-02,   2.82886744e+02,   6.79663728e+01,
#						   0.04*pc,0.,0.,0.,0.,0.],
#				  		  [0.,0.,0.,0.,0.,0.,0.,
#						   0.,0.,0.,0.,0.,0.,
#						   0.,0.,0.,0.,0.,0.,
#						   0.04*pc,0.,0.,0.,0.,0.], size=nwalkers)


# Three objects with initial decent guess
#x0 = em.utils.sample_ball([3.99298621e+01, 4.50478826e-03, -1.28610303e-02, -5.16967075e+00, 2.64503308e+01, 2.78545427e+00, 2.41850038e+22,
#						   1.51184008e+16, -1.00068439e+00, 2.22895592e+02, 4.29469497e-01, 6.70875379e+01, 3.12620663e+02,
#						   1.05126070e+17, -2.06083486e+00, 2.10324682e+02, 6.39904114e-02, 2.76476139e+02, 6.24846279e+01,
#						   0.04*pc,-1.,0.,0.,0.,0.],
#				  		  [0.1,0.0001,0.0001,20.,20.,20.,0.1*kpc,
#						   0.0005*pc,0.1,3.6,0.1,3.6,3.6,
#						   0.0004*pc,0.1,3.6,0.1,3.6,3.6,
#						   0.004*pc,0.1,0.,0.,0.,0.], size=nwalkers)

# Three objects with initial decent guess, no unused free parameters, includes variable variance
#x0 = em.utils.sample_ball([3.99298621e+01, 1.0e15, 4.50478826e-03, -1.28610303e-02, -5.16967075e+00, 2.64503308e+01, 2.78545427e+00, 2.41850038e+22,
#						   1.51184008e+16, -1.00068439e+00, 2.22895592e+02, 4.29469497e-01, 6.70875379e+01, 3.12620663e+02,
#						   1.05126070e+17, -2.06083486e+00, 2.10324682e+02, 6.39904114e-02, 2.76476139e+02, 6.24846279e+01,
#						   0.04*pc],
#				  		  [0.1,1.0e15,0.0001,0.0001,20.,20.,20.,0.1*kpc,
#						   0.0005*pc,0.1,3.6,0.1,3.6,3.6,
#						   0.0004*pc,0.1,3.6,0.1,3.6,3.6,
#						   0.004*pc], size=nwalkers)

# Three objects with initial decent guess, rp allowed to float (but not above rt), no unused free parameters
#x0 = em.utils.sample_ball([3.99298621e+01, 4.50478826e-03, -1.28610303e-02, -5.16967075e+00, 2.64503308e+01, 2.78545427e+00, 2.41850038e+22,
#						   1.51184008e+16, -1.00068439e+00, 2.22895592e+02, 4.29469497e-01, 6.70875379e+01, 3.12620663e+02,
#						   1.05126070e+17, -2.06083486e+00, 2.10324682e+02, 6.39904114e-02, 2.76476139e+02, 6.24846279e+01,
#						   0.04*pc,-3.0],
#				  		  [0.1,0.0001,0.0001,20.,20.,20.,0.1*kpc,
#						   0.0005*pc,0.1,3.6,0.1,3.6,3.6,
#						   0.0004*pc,0.1,3.6,0.1,3.6,3.6,
#						   0.04*pc,1.0], size=nwalkers)

# Three objects with initial decent guess, rp allowed to float, no unused free parameters, includes variable variance
#x0 = em.utils.sample_ball([39.9595606861, 1.e+15, 0.00336573644595, 0.00189323680646, -4.44312540168, -1.71856390183, 32.0694144443, 2.64426936195e+22,
#						   1.576992696e+16, -0.907088055947, 225.398067805, 0.631748105165, 63.0430664828, 44.6995421214,
#						   1.05126070e+17, -2.06083486e+00, 2.10324682e+02, 6.39904114e-02, 2.76476139e+02, 297.5153721,
#						   0.04*pc,-3.0],
#				  		  [0.1,1.0e15,0.0001,0.0001,20.,20.,20.,0.1*kpc,
#						   0.0005*pc,0.1,3.6,0.1,3.6,3.6,
#						   0.0004*pc,0.1,3.6,0.1,3.6,3.6,
#						   0.04*pc,1.0], size=nwalkers)

# Three objects with initial decent guess, rp allowed to float, w allowed to float, no unused free parameters, includes variable variance
#x0 = em.utils.sample_ball([39.9377048586, 1.e-7, 100., 0.00259229857886, 0.00202617827851, -3.19123397664, -5.26122862936, -12.1680808296, 2.56181937951e+22,
#						   1.55372337623e+16, -0.922512660801, 43.9219041718, 0.629817298992, 244.320447563, 45.0406570285,
#						   1.03433664987e+17, -1.78166243609, 188.411516325, 0.0785951978784, 285.378895166, 62.9907152784,
#						   0.04*pc,-2.,2.76476139e+02],
#				  		  [0.1,1.e-7,100.,0.0001,0.0001,20.,20.,20.,0.1*kpc,
#						   0.0005*pc,0.1,3.6,0.1,3.6,3.6,
#						   0.0004*pc,0.1,3.6,0.1,3.6,3.6,
#						   0.04*pc,1.0,3.6], size=nwalkers)

# Three objects with initial decent guess, rp fixed, w allowed to float, no unused free parameters, includes variable variance
#x0 = em.utils.sample_ball([39.9377048586, 1.e-7, 100., 0.00259229857886, 0.00202617827851, -3.19123397664, -5.26122862936, -12.1680808296, 2.56181937951e+22,
#						   1.55372337623e+16, -0.922512660801, 43.9219041718, 0.629817298992, 244.320447563, 45.0406570285,
#						   1.03433664987e+17, -1.78166243609, 188.411516325, 0.0785951978784, 285.378895166, 62.9907152784,
#						   0.04*pc,2.76476139e+02],
#				  		  [0.1,1.e-7,100.,0.0001,0.0001,20.,20.,20.,0.1*kpc,
#						   0.0005*pc,0.1,3.6,0.1,3.6,3.6,
#						   0.0004*pc,0.1,3.6,0.1,3.6,3.6,
#						   0.04*pc,3.6], size=nwalkers)

guess = [39.9584143435, 2.59384025953e+22]
spread = [0.1,0.1*kpc]

for i in xrange(len(types)):
	if varia[i] != -1:
		if types[i] == 'pxy':
			guess.append(1.e-7)
			spread.append(1.e-8)
		elif types[i] == 'vz':
			guess.append(100.)
			spread.append(10.)
		else:
			print "Illegal varia type"
			sys.exit(0)

if args.inputs == 0 or args.inputs == 1:
	guess.extend([0.00410019037373, -0.0118283833404, -5.88758979397, 33.8514299745, 5.04693439458])
	spread.extend([0.0001,0.0001,2.,2.,2.])
if args.inputs == 0 or args.inputs == 2:
	guess.extend([0.00188458402115, -0.000820145144759, 6.27616992363, 4.46443106758, -4.56136189986])
	spread.extend([0.0001,0.0001,2.,2.,2.])
guess.extend([1.56756036064e+16, -0.940279303786, 42.8722842131, 0.637093635944, 245.708554749, 44.6488697179,
		      9.33954553832e+16, -1.27583250582, 183.295010867, 0.0931688837147, 283.490710366, 70.4401624252,
		      1.e+18])
spread.extend([0.0005*pc,0.1,3.6,0.1,3.6,3.6,
		       0.0004*pc,0.1,3.6,0.1,3.6,3.6,
		       0.1*pc])

if not args.samerp:
	guess.append(-2.)
	spread.append(1.)

if not args.samew:
	guess.append(283.490710366)
	spread.append(36.)

x0 = em.utils.sample_ball(guess, spread, size=nwalkers)
#x0 = em.utils.sample_ball([39.9584143435, 2.59384025953e+22,
#						   7.13167564576e-10, 2.74688202102e-09, 4.49782433776e-07, 2.28807118575e-07, 218.203119924, 155.743124678,
#						   0.00410019037373, -0.0118283833404, -5.88758979397, 33.8514299745, 5.04693439458, 
#						   0.00188458402115, -0.000820145144759, 6.27616992363, 4.46443106758, -4.56136189986,
#						   1.56756036064e+16, -0.940279303786, 42.8722842131, 0.637093635944, 245.708554749, 44.6488697179,
#						   9.33954553832e+16, -1.27583250582, 183.295010867, 0.0931688837147, 283.490710366, 70.4401624252
#						   ],
#						  [0.1,0.1*kpc,1.e-9,1.e-9,1.e-7,1.e-7,10.,10.,
#						   0.0001,0.0001,2.,2.,2.,
#						   0.0001,0.0001,2.,2.,2.,
#						   0.0005*pc,0.1,3.6,0.1,3.6,3.6,
#						   0.0004*pc,0.1,3.6,0.1,3.6,3.6
#						   ], size=nwalkers)

#x0 = em.utils.sample_ball([39.9584143435, 2.59384025953e+22,
#						   7.13167564576e-10, 2.74688202102e-09, 4.49782433776e-07, 218.203119924, 155.743124678,
#						   0.00410019037373, -0.0118283833404, -5.88758979397, 33.8514299745, 5.04693439458, 
#						   0.00188458402115, -0.000820145144759, 6.27616992363, 4.46443106758, -4.56136189986,
#						   1.56756036064e+16, -0.940279303786, 42.8722842131, 0.637093635944, 245.708554749, 44.6488697179,
#						   9.33954553832e+16, -1.27583250582, 183.295010867, 0.0931688837147, 283.490710366, 70.4401624252,
#						   1.11284254908e+17, -1.27583250582],
#						  [0.1,0.1*kpc,1.e-9,1.e-9,1.e-7,10.,10.,
#						   0.0001,0.0001,2.,2.,2.,
#						   0.0001,0.0001,2.,2.,2.,
#						   0.0005*pc,0.1,3.6,0.1,3.6,3.6,
#						   0.0004*pc,0.1,3.6,0.1,3.6,3.6,
#						   0.04*pc,0.1], size=nwalkers)

#x0 = em.utils.sample_ball([39.9584143435, 2.59384025953e+22,
#						   7.13167564576e-10, 2.74688202102e-09, 4.49782433776e-07, 218.203119924, 155.743124678,
#						   0.00410019037373, -0.0118283833404, -5.88758979397, 33.8514299745, 5.04693439458, 
#						   0.00188458402115, -0.000820145144759, 6.27616992363, 4.46443106758, -4.56136189986,
#						   1.56756036064e+16, -0.940279303786, 42.8722842131, 0.637093635944, 245.708554749, 44.6488697179,
#						   9.33954553832e+16, -1.27583250582, 183.295010867, 0.0931688837147, 283.490710366, 70.4401624252,
#						   1.11284254908e+17, 283.139781117],
#						  [0.1,0.1*kpc,1.e-9,1.e-9,1.e-7,10.,10.,
#						   0.0001,0.0001,2.,2.,2.,
#						   0.0001,0.0001,2.,2.,2.,
#						   0.0005*pc,0.1,3.6,0.1,3.6,3.6,
#						   0.0004*pc,0.1,3.6,0.1,3.6,3.6,
#						   0.04*pc, 3.6], size=nwalkers)

# Forced high eccentricity
#x0 = em.utils.sample_ball([3.99298621e+01, 4.50478826e-03, -1.28610303e-02, -5.16967075e+00, 2.64503308e+01, 2.78545427e+00, 2.41850038e+22,
#						   1.53696988e+16, -9.86146230e-01, 2.21992919e+02, 4.34253060e-01, 6.70356104e+01, 3.14016468e+02,
#						   0.04*pc,-4.,360.,1.,360.,360.],
#				  		  [0.1,0.0001,0.0001,20.,20.,20.,0.1*kpc,0.0005*pc,0.1,36.,0.1,36.,36.,0.004*pc,0.0,360.,0.1,360.,360.], size=nwalkers)
# S0-2 only
#x0 = em.utils.sample_ball([3.99298621e+01, 4.50478826e-03, -1.28610303e-02, -5.16967075e+00, 2.64503308e+01, 2.78545427e+00, 2.41850038e+22,
#						   1.53696988e+16, -9.86146230e-01, 2.21992919e+02, 4.34253060e-01, 6.70356104e+01, 3.14016468e+02],
#				  		  [0.1,0.0001,0.0001,20.,20.,20.,0.1*kpc,0.0005*pc,0.1,36.,0.1,36.,36.], size=nwalkers)
#x0 = em.utils.sample_ball([40.,0.001,0.001,0.,0.,0.,8.36*kpc,0.005*pc,-1.,360.,1.,360.,360.],
#				  		  [0.1,0.0001,0.0001,20.,20.,20.,0.1*kpc,0.0005*pc,0.1,360.,0.1,360.,360.], size=nwalkers)
#x0 = em.utils.sample_ball([40.,0.001,0.001,0.,0.,0.,8.36*kpc,0.005*pc,-1.,360.,1.,360.,360.,0.04*pc,-1,360.,1.,360.,360.],
#				  		  [0.1,0.0001,0.0001,20.,20.,20.,0.1*kpc,0.0005*pc,0.1,360.,0.1,360.,360.,0.004*pc,0.1,360.,0.1,360.,360.], size=nwalkers)
#x0 = em.utils.sample_ball([40.,0.001,0.001,0.,0.,0.,0.001,0.001,0.,0.,0.,8.36*kpc,0.005*pc,-1.,360.,1.,360.,360.,0.04*pc,-1,360.,1.,360.,360.],
#				  		  [0.1,0.0001,0.0001,20.,20.,20.,0.0001,0.0001,20.,20.,20.,0.1*kpc,0.0005*pc,0.1,360.,0.1,360.,360.,0.004*pc,0.1,360.,0.1,360.,360.], size=nwalkers)
#x0 = em.utils.sample_ball([40.,0.001,0.001,0.,0.,0.001,0.001,0.,0.,0.001,0.001,0.,0.,8.36*kpc,0.,0.005*pc,-1.,360.,1.,360.,360.,0.04*pc,-1,360.,1.,360.,360.],
#				  		  [0.1,0.0001,0.0001,20.,20.,0.0001,0.0001,20.,20.,0.0001,0.0001,20.,20.,0.1*kpc,20.,0.0005*pc,0.1,360.,0.1,360.,360.,0.004*pc,0.1,360.,0.1,360.,360.], size=nwalkers)

#pos = np.empty([nwalkers, ndim])
#prob = np.empty([nwalkers])
#for i, x in enumerate(x0):
#	annealout = optimize.fmin_cg(neg_obj_func, x, args = (ptimes, vtimes), full_output = True)
#	print obj_func(x, ptimes, vtimes)
#	#annealout = optimize.anneal(neg_obj_func, x, args = (ptimes, vtimes), full_output = True, schedule = 'boltzmann')
#	print annealout
#	pos[i] = annealout[0]
#	prob[i] = -annealout[1]
#	print "Prob:", prob[i]

ndim = len(x0[0])
if (nwalkers < 2*ndim):
	print 'Error: Need walkers >= 2 * ndim'
	sys.exit(0)

pool = MPIPool()
if not pool.is_master():
	pool.wait()
	sys.exit(0)

sampler = em.EnsembleSampler(nwalkers, ndim, obj_func, args = [times, types, measurements, errors, objects, coords, varia, args.prior, 1, False], pool = pool)

# Doesn't accept args, difficult to use.
#sampler = em.PTSampler(ntemps, nwalkers, ndim, obj_func, args = [times, types, measurements, errors, objects, coords, varia], pool = pool)

pos = x0
prob = np.array([obj_func(x, times, types, measurements, errors, objects, coords, varia, args.prior, 1, False) for x in x0])
state = sampler.random_state

for t, result in enumerate(sampler.sample(pos, iterations=nsteps, storechain=False, lnprob0=prob, rstate0=state)):
	print t
	pos, prob, state = result

	temp = temp*(1.0/t0)**(1.0/nburn)
	#prob = prob/(1.0/t0)**(1.0/nsteps)
	#prob = np.array([obj_func(x, times, types, measurements, errors, objects, coords, varia, 1, False) for x in pos])

	mhx =  x[nvaria+1:5*ncoords+nvaria+1:5]
	mhy =  x[nvaria+2:5*ncoords+nvaria+1:5]
	mhvx = x[nvaria+3:5*ncoords+nvaria+1:5]
	mhvy = x[nvaria+4:5*ncoords+nvaria+1:5]
	mhvz = x[nvaria+5:5*ncoords+nvaria+1:5]
	mhz =  x[5*ncoords+nvaria+1]

	# Only works if all objects are full
	#for p, pro in enumerate(prob):
	#	for i, xx in enumerate(x[5*ncoords+3:]):
	#		mi = np.mod(i, 6)
	#		if mi == 2 or mi == 4 or mi == 5:
	#			x[5*ncoords+3+i] = np.mod(x[5*coords+3+i], 360.)

	best_probi = prob.argmax()
	best_prob = prob[best_probi]
	best_pos = pos[best_probi]
	if (t < nburn):
		replacecount = 0
		for i, p in enumerate(prob):
			#if (best_prob - p)/temp > zstretch*np.log(nwalkers + 2.0):
			#print (best_prob - p)/temp, (np.log(zstretch)*(ndim - 1.) + np.log(nwalkers))
			if (best_prob - p)/temp > (np.log(zstretch)*(ndim - 1.) + np.log(nwalkers)):
				replacecount += 1
				#print 'replacing bad walker:', i, p
				#pos[i] = perturb(best_pos, 0.01)
				ni = np.random.randint(nwalkers)
				pos[i] = pos[ni]
				prob[i] = prob[ni]
		print 'Replaced', replacecount, 'walkers (', 100.*float(replacecount)/float(nwalkers), '%)'
	print ', '.join(map(str,best_pos))
	best_y = -best_prob
	print best_prob
	best_prob = obj_func(best_pos, times, types, measurements, errors, objects, coords, varia, False, 0, True)
	print best_prob
	lmh = best_pos[0]
	mh = 10.**lmh
	semias = [[] for _ in xrange(nobjects)]
	taus = [[] for _ in xrange(nobjects)]
	periods = [[] for _ in xrange(nobjects)]
	perits = [[] for _ in xrange(nobjects)]
	for i, e in enumerate(elements):
		semias[i] = elements[i][0]
		taus[i] = elements[i][3]
		periods[i] = 2.*np.pi*np.sqrt(semias[i]**3/(G*mh))
		perits[i] = zerot + (taus[i] - 1.)*periods[i]
		print names[i], 'prev peri time:', perits[i]/yr
		print names[i], 'next peri time:', (zerot + taus[i]*periods[i])/yr
	#print 'Reduced chi^2:', -temp*prob[prob.argmax()]/(datalen - ndim - 1.)
	best_chi2 = -best_prob/(datalen - ndim - 1.)
	print 'Reduced chi^2:', best_chi2

pool.close()
dp = 0.00025
velztmin = 2000.
velztmax = 2020.
velzvmin = -4000.
velzvmax = 4000.
velzdt = 0.025
gcolors = ['b','g','y']
lgcolors = [(0.5,0.5,0.75),(0.5,0.75,0.5),(0.75,0.75,0.5)]

if pool.is_master():
	if args.dataset == 'sch':
		f = open('sch.scores', 'a', os.O_NONBLOCK)
		f.write(str(args.id) + ' ' + str(best_y) + ' ' + str(best_chi2) + '\n')
		f.flush()
		fname = 'pos.sch'+str(args.id)+'.out'
	elif args.dataset == 'yel':
		f = open('yel.scores', 'a', os.O_NONBLOCK)
		f.write(str(args.id) + ' ' + str(best_y) + ' ' + str(best_chi2) + '\n')
		f.flush()
		fname = 'pos.yel'+str(args.id)+'.out'
	elif args.dataset == 'lu':
		f = open('lu.scores', 'a', os.O_NONBLOCK)
		f.write(str(args.id) + ' ' + str(best_y) + ' ' + str(best_chi2) + '\n')
		f.flush()
		fname = 'pos.lu'+str(args.id)+'.out'
	else:
		fname = 'pos.out'
	np.savetxt(fname, np.array([nwalkers,ndim,nsteps]))
	f_handle = file(fname, 'a')
	np.savetxt(f_handle, pos)
	np.savetxt(f_handle, prob)
	f_handle.close()

	obj_func(pos[prob.argmax()], times, types, measurements, errors, objects, coords, varia, False, 0, True)

	orbtimes = [[] for _ in xrange(nobjects)]
	orbtimes2 = [[] for _ in xrange(nobjects)]
	orbpos = [[] for _ in xrange(nobjects)]
	orbvel = [[] for _ in xrange(nobjects)]
	kes = get_star_ke(elements, lmh)
	for i, e in enumerate(elements):
		orbtimes[i] = np.arange(0., 1. + dp, dp)*periods[i] + taus[i]*periods[i]
		orbpos[i] = kes[i].xyzPos(orbtimes[i])*impc
		orbtimes[i] = (perits[i] + orbtimes[i] - taus[i]*periods[i])/yr

		orbtimes2[i] = np.arange(0., 4. + dp, dp)*periods[i] + taus[i]*periods[i]
		orbvel[i] = -kes[i].xyzVel(orbtimes2[i])*ikm
		orbtimes2[i] = (perits[i] + orbtimes2[i] - taus[i]*periods[i])/yr

	posx = [[] for _ in xrange(nobjects)]
	posy = [[] for _ in xrange(nobjects)]
	velx = [[] for _ in xrange(nobjects)]
	vely = [[] for _ in xrange(nobjects)]
	velz = [[] for _ in xrange(nobjects)]
	pltt = [[] for _ in xrange(nobjects)]
	for t, stimes in enumerate(times):
		gi = objects[t]
		ci = coords[t]
		if types[t] == 'pxy':
			posx[gi].extend((vecs[t][:,0] - 2.*mhz*np.tan(mhx[ci]*iasec) - mhvx[ci]*km*stimes)*impc)
			posy[gi].extend((vecs[t][:,1] - 2.*mhz*np.tan(mhy[ci]*iasec) - mhvy[ci]*km*stimes)*impc)
		elif types[t] =='vxy':
			pltt[gi].extend((zerot + stimes)/yr)
			velx[gi].extend(vecs[t][:,0]*ikm - mhvx[ci])
			vely[gi].extend(vecs[t][:,1]*ikm - mhvy[ci])
		elif types[t] =='vz':
			pltt[gi].extend((zerot + stimes)/yr)
			velz[gi].extend(-(vecs[t]*ikm - mhvz[ci])) # Flip sign back to match convention

	#Now do ensemble stuff
	print 'Assembling ensemble'
	ensembletimes = np.arange(velztmin, velztmax + velzdt, velzdt)
	ensemblevels = np.zeros(shape=(nwalkers,nobjects,len(ensembletimes),3))
	for w in xrange(nwalkers):
		obj_func(pos[w], times, types, measurements, errors, objects, coords, varia, False, 2, False)
		lmh = pos[w,0]
		kes = get_star_ke(elements, lmh)
		for i, e in enumerate(elements):
			ensemblevels[w,i,:,:] = -kes[i].xyzVel(ensembletimes*yr - zerot)*ikm

	velzminvec = np.zeros(shape=(nobjects,len(ensembletimes)))
	velzmaxvec = np.zeros(shape=(nobjects,len(ensembletimes)))
	for t in xrange(len(ensembletimes)):
		for g in xrange(nobjects):
			ensemblevels[:,g,t,2] = np.sort(ensemblevels[:,g,t,2])
			nsamp = nwalkers - np.isnan(ensemblevels[:,g,t,2]).sum()
			msig = int(nsamp - ma.floor(nsamp*ma.erf(2./np.sqrt(2.)))) #2 sigma
			psig = int(nsamp - msig)
			velzminvec[g,t] = ensemblevels[msig,g,t,2]
			velzmaxvec[g,t] = ensemblevels[psig,g,t,2]
			print msig, psig, ensemblevels[msig,g,t,2], ensemblevels[psig,g,t,2]
	print 'Done with ensemble'

	# Set globals back to best solution
	obj_func(pos[prob.argmax()], times, types, measurements, errors, objects, coords, varia, False, 0, True)

	mpl.rcParams.update({'font.size': 16})

	fig, (posplt, velzplt, velxyplt) = plt.subplots(1,3)
	#fig, (posplt, velzplt) = plt.subplots(1,2)

	fig.tight_layout()

	for g in xrange(nobjects):
		velzplt.fill_between(ensembletimes, velzminvec[g], velzmaxvec[g], facecolor=lgcolors[g], edgecolor='none', interpolate=True)

	for g in xrange(nobjects):
		#posx = [xx for (yy,xx) in sorted(zip(pltt,posx))]
		#posy = [xx for (yy,xx) in sorted(zip(pltt,posy))]
		posplt.plot(posx[g], posy[g], gcolors[g]+'o')
		posplt.plot(orbpos[g][:,0], orbpos[g][:,1], gcolors[g]+'-')

		if (g <= 1):
			#velz = [xx for (yy,xx) in sorted(zip(pltt,velz))]
			#pltt = [xx for (yy,xx) in sorted(zip(pltt,pltt))]
			velzplt.plot(pltt[g], velz[g], gcolors[g]+'o')

		velzplt.plot(orbtimes2[g], orbvel[g][:,2], gcolors[g]+'-')
		#np.set_printoptions(threshold='nan')
		#print orbtimes[i], orbvel[g][:,2]

		if (g == 2):
			velxyplt.plot(velx[g], vely[g], gcolors[g]+'o')

	velzplt.set_xlim(velztmin, velztmax)
	velzplt.set_xticks(np.arange(velztmin, velztmax, 5))
	velzplt.set_ylim(velzvmin, velzvmax)
	minorLocator = MultipleLocator(1)
	velzplt.xaxis.set_minor_locator(minorLocator)
	minorLocator = MultipleLocator(200)
	velzplt.yaxis.set_minor_locator(minorLocator)
	#velzplt.ylim(min(list(flatten(velz[0:2]))), max(list(flatten(velz[0:2]))))
	velzplt.set_xlabel('$t$ (yr)', fontsize=18)
	velzplt.set_ylabel('$v_r$ (km/s)', fontsize=18)

	posplt.invert_xaxis()
	posplt.set_xlabel('$x$ (mpc)', fontsize=18)
	posplt.set_ylabel('$y$ (mpc)', fontsize=18)

	velxyplt.set_xlabel('$v_x$ (km/s)', fontsize=18)
	velxyplt.set_ylabel('$v_y$ (km/s)', fontsize=18)

	posplt.plot(0., 0., 'ko')

	for m, mea in enumerate(measurements):
		tim = times[m] + zerot
		typ = types[m]
		err = errors[m]
		obj = objects[m]
		coo = coords[m]
		nam = names[obj]
		var = varia[m]

		if typ == 'pxy':
			x = (mhz*mea[:,0] - 2.*mhz*np.tan(mhx[coo]*iasec) - mhvx[coo]*km*(tim - zerot))*impc
			y = (mhz*mea[:,1] - 2.*mhz*np.tan(mhy[coo]*iasec) - mhvy[coo]*km*(tim - zerot))*impc

			if var != -1:
				variancepxy2 = variances[var]**2
			else:
				variancepxy2 = 0.

			u = mhz*np.sqrt(err[:,0]**2 + variancepxy2)*impc
			v = mhz*np.sqrt(err[:,1]**2 + variancepxy2)*impc

			posplt.errorbar(x, y, xerr=u, yerr=v, elinewidth=1, linewidth=0, fmt=gcolors[obj])
		elif typ == 'vz':
			y = -(mea[:,0]*ikm - mhvz[coo]) # Flip sign back to match convention

			if var != -1:
				variancevz2 = variances[var]**2
			else:
				variancevz2 = 0.

			v = np.sqrt((err[:,0]*ikm)**2 + variancevz2)

			velzplt.errorbar(tim/yr, y, yerr=v, elinewidth=1, linewidth=0, fmt=gcolors[obj])
		elif typ == 'vxy':
			#pass
			x = mhz*mea[:,0]*ikm - mhvx[coo]
			y = mhz*mea[:,1]*ikm - mhvy[coo]
			u = mhz*err[:,0]*ikm
			v = mhz*err[:,1]*ikm

			velxyplt.errorbar(x, y, xerr=u, yerr=v, elinewidth=1, linewidth=0, fmt=gcolors[obj])
		else:
			print 'Illegal measurement type'
			sys.exit(0)

	#fig.set_size_inches(20.,5.)
	#plt.savefig('fit.png',dpi=100,bbox_inches='tight')
	fig.set_size_inches(20.,5.)
	if args.dataset == 'sch':
		plt.savefig('fit.sch'+str(args.id)+'.pdf',dpi=100,bbox_inches='tight')
	elif args.dataset == 'yel':
		plt.savefig('fit.yel'+str(args.id)+'.pdf',dpi=100,bbox_inches='tight')
	elif args.dataset == 'lu':
		plt.savefig('fit.lu'+str(args.id)+'.pdf',dpi=100,bbox_inches='tight')
	else:
		plt.savefig('fit.pdf',dpi=100,bbox_inches='tight')
	#plt.show()

sys.exit()
