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
parser.add_argument('--samerp',           dest='samerp',    help='G2 and Candidate forced to have same rp', 				 default=False,      action='store_true')
parser.add_argument('--samew',            dest='samew',     help='G2 and Candidate forced to have same w',  				 default=False,      action='store_true')
parser.add_argument('--noproprec',        dest='noproprec', help='Candidate not allowed to precess prograde relative to G2', default=False,      action='store_true')
parser.add_argument('--preclim',          dest='preclim',   help='Candidate cannot precess more than this many degrees',     default=-1.,        type=float)
parser.add_argument('--prior',            dest='prior',     help='Use prior for Sgr A* properties',                          default=False,      action='store_true')
parser.add_argument('--dataset',          dest='dataset',   help='Dataset name',                                             default='',         type=str)
parser.add_argument('--id',               dest='id',        help='ID of star in dataset',  				                     default=-1,         type=int)
parser.add_argument('--nwalkers',         dest='nwalkers',  help='Number of walkers',  				                         default=-1,         type=int)
parser.add_argument('--nsteps',           dest='nsteps',    help='Number of steps',  				                         default=-1,         type=int)
parser.add_argument('--inputs',           dest='inputs',    help='Which data to use for S2/G2',  			                 default=-1,         type=int)
parser.add_argument('--units',            dest='units',     help='Output position/proper motion in physical/angular units',  default='physical', type=str)
parser.add_argument('--label',            dest='label',     help='Additional filename label',                                default='',         type=str)
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
		a = abs(e[0])
		mh = 10.**lmh
		loce = 1. - 10.**min(e[1],0.)
		period = 2.*np.pi*np.sqrt(a**3/(G*mh))
		loctau = max(min(e[3],1.),0.)*period
		ret[i] = pyasl.KeplerEllipse(a, period, e=loce, Omega=e[2], tau=loctau, w=e[4], i=e[5])

	return ret

def chunks(l, n):
	return np.array([l[i:i+n] for i in range(0, len(l), n)])

def prior_func(x):
	return 0.

def obj_func(x, times, types, measurements, errors, objects, coords, varia, prior, mode, save):
	global vecs, vecs2, temp, elements, variances, kes
	lmh =  x[0]
	mhz =  x[1]
	variances = x[2:nvaria+2]

	mhx =  x[nvaria+2:5*ncoords+nvaria+2:5]
	mhy =  x[nvaria+3:5*ncoords+nvaria+2:5]
	mhvx = x[nvaria+4:5*ncoords+nvaria+2:5]
	mhvy = x[nvaria+5:5*ncoords+nvaria+2:5]
	mhvz = x[nvaria+6:5*ncoords+nvaria+2:5]
	elements = chunks(x[5*ncoords+nvaria+2:], 6)

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

	mh = 10.**lmh

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
			rp1 = a1*10.**elements[1][1]
			rp2 = a2*10.**elements[2][1]
			if rp2 < rp1 or rp2 > rpmax: return float("-inf")
		if elements[2][1] > 0.: return float("-inf")
		elements[2][2] = elements[1][2]
		#elements[2][3] = elements[1][3]*period1/period2
		elements[2][3] = np.mod(g2reftime - np.mod(g2reftime - elements[1][3]*period1, period1), period2)/period2
		#elements[2][3] = (1. - (1.-elements[1][3])*period1/period2)
		if args.samew:
			elements[2][4] = elements[1][4]
		else:
			if args.noproprec:
				# Retrograde precession only, GR won't be important
				if elements[2][4] > elements[1][4]: return float("-inf")
			if args.preclim > 0.:
				if np.abs(elements[2][4] - elements[1][4]) > args.preclim: return float("-inf")
		elements[2][5] = elements[1][5]

	kes = get_star_ke(elements, lmh)

	# All globals except vecs/vecs2 should be calculated before premature returns
	if min(variances) < 0: return float("-inf")

	for i, e in enumerate(elements):
		if e[0] < 1.e13 or e[0] > 1.e20: return float("-inf")
		if e[1] > 0.: return float("-inf")
		if e[2] > 360. or e[2] < 0.: return float("-inf")
		if e[3] < 0. or e[3] > 1.: return float("-inf")
		if e[4] > 360. or e[4] < 0.: return float("-inf")
		if e[5] > 360. or e[5] < 0.: return float("-inf")

	# Skip obj func calculation in this mode
	if mode == 2:
		return float("-inf")

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

global temp, elements, vecs, vecs2, variances, kes

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
masec = 1.e3*asec
imasec = 1./masec

temp = t0

zscale = 2.
zstretch = ((zscale - 1.) + 1.)**2/zscale
	
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
	canddata = np.loadtxt(cd+"schoedel-2009/Sch"+str(args.id)+".points")
	candvxydata = np.loadtxt(cd+"schoedel-2009/Sch"+str(args.id)+".vxy")
	candcoorxy = 1
	candcoorz  = -1
elif args.dataset == 'yel':
	if args.id == -1:
		args.id = 20
	canddata = np.loadtxt(cd+"yelda-2010/Yelda"+str(args.id)+".points")
	candvxydata = np.loadtxt(cd+"yelda-2010/Yelda"+str(args.id)+".vxy")
	candcoorxy = 0
	candcoorz  = -1
elif args.dataset == 'lu':
	if args.id == -1:
		args.id = 1
	canddata = np.loadtxt(cd+"lu-2009/Lu"+str(args.id)+".points")
	candvxydata = np.loadtxt(cd+"lu-2009/Lu"+str(args.id)+".vxy")
	candvdata = np.loadtxt(cd+"lu-2009/Lu"+str(args.id)+".rv")
	candcoorxy = 0
	candcoorz  = 0
elif args.dataset == 'do':
	if args.id == -1:
		args.id = 1
	canddata = np.loadtxt(cd+"do-2013/Do"+str(args.id)+".points")
	candvxydata = np.loadtxt(cd+"do-2013/Do"+str(args.id)+".vxy")
	candvdata = np.loadtxt(cd+"do-2013/Do"+str(args.id)+".rv")
	candcoorxy = 0
	candcoorz  = 0
elif args.dataset == 'unp':
	if args.id == 1:
		canddata = np.loadtxt(cd+"unpublished/S259.points")
		candvxydata = np.loadtxt(cd+"unpublished/S259.vxy")
		candvdata = np.loadtxt(cd+"unpublished/S259.rv")
		candcoorxy = 1
		candcoorz  = 1
	elif args.id == 2:
		canddata = np.loadtxt(cd+"yelda-2010/Yelda126.points")
		candvxydata = np.loadtxt(cd+"yelda-2010/Yelda126.vxy")
		candvdata = np.loadtxt(cd+"unpublished/S2-84.rv")
		candcoorxy = 0
		candcoorz  = 1
	elif args.id == 3:
		canddata = np.loadtxt(cd+"yelda-2010/Yelda191.points")
		candvxydata = np.loadtxt(cd+"yelda-2010/Yelda191.vxy")
		candvdata = np.loadtxt(cd+"unpublished/S3-223.rv")
		candcoorxy = 0
		candcoorz  = 1
	elif args.id == 4:
		canddata = np.loadtxt(cd+"yelda-2010/Yelda230.points")
		candvxydata = np.loadtxt(cd+"yelda-2010/Yelda230.vxy")
		candvdata = np.loadtxt(cd+"unpublished/S4-23.rv")
		candcoorxy = 0
		candcoorz  = 1
	elif args.id == 5:
		canddata = np.loadtxt(cd+"unpublished/S3-223.points")
		candvxydata = np.loadtxt(cd+"unpublished/S3-223.vxy")
		candvdata = np.loadtxt(cd+"unpublished/S3-223.rv")
		candcoorxy = 1
		candcoorz  = 1
	elif args.id == 6:
		canddata = np.loadtxt(cd+"unpublished/S2-84.points")
		candvxydata = np.loadtxt(cd+"unpublished/S2-84.vxy")
		candvdata = np.loadtxt(cd+"unpublished/S2-84.rv")
		candcoorxy = 1
		candcoorz  = 1
	elif args.id == 7:
		canddata = np.loadtxt(cd+"unpublished/S182.points")
		candvxydata = np.loadtxt(cd+"unpublished/S182.vxy")
		candcoorxy = 1
		candcoorz  = -1
	elif args.id == 8:
		canddata = np.loadtxt(cd+"unpublished/S3-29.points")
		candvxydata = np.loadtxt(cd+"unpublished/S3-29.vxy")
		candvdata = np.loadtxt(cd+"unpublished/S3-29.rv")
		candcoorxy = 1
		candcoorz  = 1
	elif args.id == 9:
		canddata = np.loadtxt(cd+"unpublished/S2-198.points")
		candvxydata = np.loadtxt(cd+"unpublished/S2-198.vxy")
		candcoorxy = 1
		candcoorz  = -1
	else:
		print "Undefined ID for dataset."
		sys.exit(0)
else:
	print "Invalid dataset selected"
	sys.exit(0)

print "Fitting #" + str(args.id) + " in " + args.dataset + " dataset."

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
if candcoorz != -1:
	candvdata = np.reshape(candvdata, (-1, 3))
	candvdata[:,0] = candvdata[:,0]*yr
	candvdata[:,1] = -candvdata[:,1]
	candvdata[:,1:] = candvdata[:,1:]*km

# Both datasets

if args.inputs == 0:
	if candcoorz != -1:
		times = np.array([s2data[:,0],s2data2[:,0],g2data[:,0],g2data2[:,0],canddata[:,0],s2vdata[:,0],s2vdata2[:,0],g2vdata[:,0],g2vdata2[:,0],candvxydata[:,0],candvdata[:,0]])
		types = ['pxy','pxy','pxy','pxy','pxy','vz','vz','vz','vz','vxy','vz']
		measurements = [s2data[:,1:3],s2data2[:,1:3],g2data[:,1:3],g2data2[:,1:3],canddata[:,1:3],s2vdata[:,1:2],s2vdata2[:,1:2],g2vdata[:,1:2],g2vdata2[:,1:2],candvxydata[:,1:3],candvdata[:,1:2]]
		errors = [s2data[:,3:5],s2data2[:,3:5],g2data[:,3:5],g2data2[:,3:5],canddata[:,3:5],s2vdata[:,2:3],s2vdata2[:,2:3],g2vdata[:,2:3],g2vdata2[:,2:3],candvxydata[:,3:5],candvdata[:,2:3]]
		objects = [0,0,1,1,2,0,0,1,1,2,2]
		coords = [0,1,0,1,candcoorxy,0,1,0,1,0,candcoorz]
		kind = [0,0,1]
		names = ['S2','G2','Candidate']
		varia = [0,1,2,3,-1,-1,-1,4,4,-1,-1]
	else:
		times = np.array([s2data[:,0],s2data2[:,0],g2data[:,0],g2data2[:,0],canddata[:,0],s2vdata[:,0],s2vdata2[:,0],g2vdata[:,0],g2vdata2[:,0],candvxydata[:,0]])
		types = ['pxy','pxy','pxy','pxy','pxy','vz','vz','vz','vz','vxy']
		measurements = [s2data[:,1:3],s2data2[:,1:3],g2data[:,1:3],g2data2[:,1:3],canddata[:,1:3],s2vdata[:,1:2],s2vdata2[:,1:2],g2vdata[:,1:2],g2vdata2[:,1:2],candvxydata[:,1:3]]
		errors = [s2data[:,3:5],s2data2[:,3:5],g2data[:,3:5],g2data2[:,3:5],canddata[:,3:5],s2vdata[:,2:3],s2vdata2[:,2:3],g2vdata[:,2:3],g2vdata2[:,2:3],candvxydata[:,3:5]]
		objects = [0,0,1,1,2,0,0,1,1,2]
		coords = [0,1,0,1,candcoorxy,0,1,0,1,0]
		kind = [0,0,1]
		names = ['S2','G2','Candidate']
		varia = [0,1,2,3,-1,-1,-1,4,4,-1]
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

guess = [39.9584143435, 2.59384025953e+22]
spread = [0.1,0.1*kpc]

# Currently, varia must be numbered in sequential order (e.g. 1,2,2,3,3,4).
varcnt = -1
for i in xrange(len(types)):
	if varia[i] != -1 and varia[i] != varcnt:
		varcnt = varcnt + 1
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
		      0.1*pc])
spread.extend([0.0005*pc,0.1,3.6,0.1,3.6,3.6,
		       0.0004*pc,0.1,10.,0.1,10.,10.,
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

#ntemps = 8
#sampler = em.PTSampler(ntemps, nwalkers, ndim, obj_func, prior_func, loglargs = [times, types, measurements, errors, objects, coords, varia, args.prior, 1, False], pool = pool)

pos = x0
prob = np.array([obj_func(x, times, types, measurements, errors, objects, coords, varia, args.prior, 1, False) for x in x0])
state = sampler.random_state

alltime_best_y = float("inf")
alltime_best_chi2 = float("inf")
alltime_best_pos = x0[0]
for t, result in enumerate(sampler.sample(pos, iterations=nsteps, storechain=False, lnprob0=prob, rstate0=state)):
	print t
	pos, prob, state = result

	temp = temp*(1.0/t0)**(1.0/nburn)

	best_probi = prob.argmax()
	best_y = -prob[best_probi]
	best_pos = pos[best_probi]
	best_chi2 = -obj_func(best_pos, times, types, measurements, errors, objects, coords, varia, False, 0, True)/(datalen - ndim - 1.)

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
		print names[i] + ' prev peri time: ', str(perits[i]/yr) + ', next peri time: ' + str((zerot + taus[i]*periods[i])/yr)

	print 'Current best y:  ' + str(best_y) + ', with reduced chi^2: ' + str(best_chi2)
	if best_y < alltime_best_y:
		alltime_best_y = best_y
		alltime_best_chi2 = best_chi2
		alltime_best_pos = best_pos
	print 'All-time best y: ' + str(alltime_best_y) + ', with reduced chi^2: ' + str(alltime_best_chi2)

	if t < nburn:
		replacecount = 0
		for i, p in enumerate(prob):
			if (-alltime_best_y - p)/temp > (np.log(zstretch)*(ndim - 1.) + np.log(nwalkers)):
				replacecount += 1
				#print 'replacing bad walker:', i, p
				#coin = np.random.rand()
				#if coin > 0.5 or t == 1:
				ni = np.random.randint(nwalkers)
				pos[i] = pos[ni]
				prob[i] = prob[ni]
				#else:
				#	# Not recalculating prob, but very tiny perturbations.
				#	#pos[i] = perturb(alltime_best_pos, 0.0001)
				#	pos[i] = alltime_best_pos
				#	prob[i] = -alltime_best_y
		print 'Replaced', replacecount, 'walkers (', 100.*float(replacecount)/float(nwalkers), '%)'
	np.savetxt(sys.stdout, best_pos[None], '%.3e')

pool.close()
dp = 0.00025
velztmin = 2000.
velztmax = 2020.
velzvmin = -4000.
velzvmax = 4000.
velzdt = 0.01
gcolors = ['b','g','y']
lgcolors = [(0.5,0.5,0.75),(0.5,0.75,0.5),(0.75,0.75,0.5)]

if args.label != '':
	label = '.'+args.label;
else:
	label = ''

if pool.is_master():
	if args.dataset == 'sch':
		f = open('sch'+label+'.scores', 'a', os.O_NONBLOCK)
		f.write(str(args.id) + ' ' + str(alltime_best_y) + ' ' + str(alltime_best_chi2) + '\n')
		f.flush()
		fname = 'pos.sch'+str(args.id)+label+'.out'
	elif args.dataset == 'yel':
		f = open('yel'+label+'.scores', 'a', os.O_NONBLOCK)
		f.write(str(args.id) + ' ' + str(alltime_best_y) + ' ' + str(alltime_best_chi2) + '\n')
		f.flush()
		fname = 'pos.yel'+str(args.id)+label+'.out'
	elif args.dataset == 'lu':
		f = open('lu'+label+'.scores', 'a', os.O_NONBLOCK)
		f.write(str(args.id) + ' ' + str(alltime_best_y) + ' ' + str(alltime_best_chi2) + '\n')
		f.flush()
		fname = 'pos.lu'+str(args.id)+label+'.out'
	elif args.dataset == 'do':
		f = open('do'+label+'.scores', 'a', os.O_NONBLOCK)
		f.write(str(args.id) + ' ' + str(alltime_best_y) + ' ' + str(alltime_best_chi2) + '\n')
		f.flush()
		fname = 'pos.do'+str(args.id)+label+'.out'
	elif args.dataset == 'unp':
		f = open('unp'+label+'.scores', 'a', os.O_NONBLOCK)
		f.write(str(args.id) + ' ' + str(alltime_best_y) + ' ' + str(alltime_best_chi2) + '\n')
		f.flush()
		fname = 'pos.unp'+str(args.id)+label+'.out'
	else:
		fname = 'pos.out'
	np.savetxt(fname, np.array([nwalkers,ndim,nsteps]))
	f_handle = file(fname, 'a')
	np.savetxt(f_handle, pos)
	np.savetxt(f_handle, prob)
	f_handle.close()

	#obj_func(pos[prob.argmax()], times, types, measurements, errors, objects, coords, varia, False, 0, True)
	obj_func(alltime_best_pos, times, types, measurements, errors, objects, coords, varia, False, 0, True)

	# Save all-time best info
	at_lmh  = alltime_best_pos[0]
	at_mhz  = alltime_best_pos[1]
	at_mhx  = alltime_best_pos[nvaria+2:5*ncoords+nvaria+2:5]
	at_mhy  = alltime_best_pos[nvaria+3:5*ncoords+nvaria+2:5]
	at_mhvx = alltime_best_pos[nvaria+4:5*ncoords+nvaria+2:5]
	at_mhvy = alltime_best_pos[nvaria+5:5*ncoords+nvaria+2:5]
	at_mhvz = alltime_best_pos[nvaria+6:5*ncoords+nvaria+2:5]
	at_elements = elements
	at_variances = variances
	at_vecs = vecs
	at_mh = 10.**at_lmh
	at_kes = get_star_ke(at_elements, at_lmh)

	# Generate full orbital paths
	orbtimes = [[] for _ in xrange(nobjects)]
	orbtimes2 = [[] for _ in xrange(nobjects)]
	orbpos = [[] for _ in xrange(nobjects)]
	orbvel = [[] for _ in xrange(nobjects)]
	semias = [[] for _ in xrange(nobjects)]
	taus = [[] for _ in xrange(nobjects)]
	periods = [[] for _ in xrange(nobjects)]
	perits = [[] for _ in xrange(nobjects)]
	for i, e in enumerate(at_elements):
		semias[i] = at_elements[i][0]
		taus[i] = at_elements[i][3]
		periods[i] = 2.*np.pi*np.sqrt(semias[i]**3/(G*at_mh))
		perits[i] = zerot + (taus[i] - 1.)*periods[i]

	for i, e in enumerate(at_elements):
		orbtimes[i] = np.arange(0., 1. + dp, dp)*periods[i] + taus[i]*periods[i]
		orbpos[i] = at_kes[i].xyzPos(orbtimes[i])*impc
		orbtimes[i] = (perits[i] + orbtimes[i] - taus[i]*periods[i])/yr

		orbtimes2[i] = np.arange(0., 4. + dp, dp)*periods[i] + taus[i]*periods[i]
		orbvel[i] = -at_kes[i].xyzVel(orbtimes2[i])*ikm
		orbtimes2[i] = (perits[i] + orbtimes2[i] - taus[i]*periods[i])/yr

	posx = [[] for _ in xrange(nobjects)]
	posy = [[] for _ in xrange(nobjects)]
	velx = [[] for _ in xrange(nobjects)]
	vely = [[] for _ in xrange(nobjects)]
	velz = [[] for _ in xrange(nobjects)]
	vzt = [[] for _ in xrange(nobjects)]
	for t, stimes in enumerate(times):
		gi = objects[t]
		ci = coords[t]
		if types[t] == 'pxy':
			posx[gi].extend((at_vecs[t][:,0] - 2.*at_mhz*np.tan(at_mhx[ci]*iasec) - at_mhvx[ci]*km*stimes)*impc)
			posy[gi].extend((at_vecs[t][:,1] - 2.*at_mhz*np.tan(at_mhy[ci]*iasec) - at_mhvy[ci]*km*stimes)*impc)
		elif types[t] =='vxy':
			velx[gi].extend(at_vecs[t][:,0]*ikm - at_mhvx[ci])
			vely[gi].extend(at_vecs[t][:,1]*ikm - at_mhvy[ci])
		elif types[t] =='vz':
			vzt[gi].extend((zerot + stimes)/yr)
			velz[gi].extend(-(at_vecs[t]*ikm - at_mhvz[ci])) # Flip sign back to match convention

	#Now do ensemble stuff
	print 'Assembling ensemble'
	ensembletimes = np.arange(velztmin, velztmax + velzdt, velzdt)
	ensemblevels = np.zeros(shape=(nwalkers,nobjects,len(ensembletimes),3))
	for w in xrange(nwalkers):
		obj_func(pos[w], times, types, measurements, errors, objects, coords, varia, False, 2, False)
		for i, k in enumerate(kes):
			ensemblevels[w,i,:,:] = -k.xyzVel(ensembletimes*yr - zerot)*ikm

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
	print 'Done with ensemble'

	mpl.rcParams.update({'font.size': 16})

	fig, (posplt, velzplt, velxyplt) = plt.subplots(1,3)
	#fig, (posplt, velzplt) = plt.subplots(1,2)

	fig.tight_layout()

	for g in xrange(nobjects):
		velzplt.fill_between(ensembletimes, velzminvec[g], velzmaxvec[g], facecolor=lgcolors[g], edgecolor='none', interpolate=True)

	for g in xrange(nobjects):
		if args.units == 'physical':
			posplt.plot(posx[g], posy[g], gcolors[g]+'o', markersize=4)
			posplt.plot(orbpos[g][:,0], orbpos[g][:,1], gcolors[g]+'-')
		else:
			posplt.plot(np.arctan(0.5*np.array(posx[g])*mpc/at_mhz)*asec, np.arctan(0.5*np.array(posy[g])*mpc/at_mhz)*asec, gcolors[g]+'o', markersize=4)
			posplt.plot(np.arctan(0.5*orbpos[g][:,0]*mpc/at_mhz)*asec, np.arctan(0.5*orbpos[g][:,1]*mpc/at_mhz)*asec, gcolors[g]+'-')

		if g <= 1 or candcoorz != -1:
			#velz = [xx for (yy,xx) in sorted(zip(vzt,velz))]
			#vzt = [xx for (yy,xx) in sorted(zip(vzt,vzt))]
			velzplt.plot(vzt[g], velz[g], gcolors[g]+'o', markersize=4)

		velzplt.plot(orbtimes2[g], orbvel[g][:,2], gcolors[g]+'-')
		#np.set_printoptions(threshold='nan')
		#print orbtimes[i], orbvel[g][:,2]

		if g == 2:
			if args.units == 'physical':
				velxyplt.plot(velx[g], vely[g], gcolors[g]+'o', markersize=4)
			else:
				velxyplt.plot(np.arctan(0.5*np.array(velx[g])*km*yr/at_mhz)*masec, np.arctan(0.5*np.array(vely[g])*km*yr/at_mhz)*masec, gcolors[g]+'o')

	velzplt.set_xlim(velztmin, velztmax)
	velzplt.set_xticks(np.arange(velztmin, velztmax, 5))
	velzplt.set_ylim(velzvmin, velzvmax)
	minorLocator = MultipleLocator(1)
	velzplt.xaxis.set_minor_locator(minorLocator)
	minorLocator = MultipleLocator(200)
	velzplt.yaxis.set_minor_locator(minorLocator)
	#velzplt.ylim(min(list(flatten(velz[0:2]))), max(list(flatten(velz[0:2]))))
	velzplt.set_xlabel('$t$ (yr)', fontsize=18)
	velzplt.set_ylabel('$v_r$ (km s$^{-1}$)', fontsize=18)

	posplt.invert_xaxis()
	if args.units == 'physical':
		posplt.set_xlabel('$x$ (mpc)', fontsize=18)
		posplt.set_ylabel('$y$ (mpc)', fontsize=18)
		velxyplt.set_xlabel('$v_x$ (km s$^{-1}$)', fontsize=18)
		velxyplt.set_ylabel('$v_y$ (km s$^{-1}$)', fontsize=18)
	else:
		posplt.set_xlabel('R.A. (arcsec)', fontsize=18)
		posplt.set_ylabel('Dec. (arcsec)', fontsize=18)
		velxyplt.set_xlabel('R.A. Prop. Mot. (milliarcsec yr$^{-1}$)', fontsize=18)
		velxyplt.set_ylabel('Dec. Prop. Mot. (milliarcsec yr$^{-1}$)', fontsize=18)

	posplt.plot(0., 0., 'ko')

	posplt.margins(0.1, tight=False)
	velxyplt.margins(0.1, tight=False)

	for m, mea in enumerate(measurements):
		tim = times[m] + zerot
		typ = types[m]
		err = errors[m]
		obj = objects[m]
		coo = coords[m]
		nam = names[obj]
		var = varia[m]

		if typ == 'pxy':
			x = (at_mhz*mea[:,0] - 2.*at_mhz*np.tan(at_mhx[coo]*iasec) - at_mhvx[coo]*km*(tim - zerot))*impc
			y = (at_mhz*mea[:,1] - 2.*at_mhz*np.tan(at_mhy[coo]*iasec) - at_mhvy[coo]*km*(tim - zerot))*impc

			if var != -1:
				variancepxy2 = at_variances[var]**2
			else:
				variancepxy2 = 0.

			u = at_mhz*np.sqrt(err[:,0]**2 + variancepxy2)*impc
			v = at_mhz*np.sqrt(err[:,1]**2 + variancepxy2)*impc

			if args.units == 'physical':
				posplt.errorbar(x, y, xerr=u, yerr=v, elinewidth=1, linewidth=0, fmt=gcolors[obj])
			else:
				posplt.errorbar(np.arctan(0.5*x*mpc/at_mhz)*asec, np.arctan(0.5*y*mpc/at_mhz)*asec,
								xerr=np.arctan(0.5*u*mpc/at_mhz)*asec, yerr=np.arctan(0.5*v*mpc/at_mhz)*asec, elinewidth=1, linewidth=0, fmt=gcolors[obj])
		elif typ == 'vz':
			y = -(mea[:,0]*ikm - at_mhvz[coo]) # Flip sign back to match convention

			if var != -1:
				variancevz2 = at_variances[var]**2
			else:
				variancevz2 = 0.

			v = np.sqrt((err[:,0]*ikm)**2 + variancevz2)

			velzplt.errorbar(tim/yr, y, yerr=v, elinewidth=1, linewidth=0, fmt=gcolors[obj])
		elif typ == 'vxy':
			#pass
			x = at_mhz*mea[:,0]*ikm - at_mhvx[coo]
			y = at_mhz*mea[:,1]*ikm - at_mhvy[coo]
			u = at_mhz*err[:,0]*ikm
			v = at_mhz*err[:,1]*ikm

			if args.units == 'physical':
				velxyplt.errorbar(x, y, xerr=u, yerr=v, elinewidth=1, linewidth=0, fmt=gcolors[obj])
			else:
				velxyplt.errorbar(np.arctan(0.5*x*km*yr/at_mhz)*masec, np.arctan(0.5*y*km*yr/at_mhz)*masec,
							      xerr=np.arctan(0.5*u*km*yr/at_mhz)*masec, yerr=np.arctan(0.5*v*km*yr/at_mhz)*masec,
								  elinewidth=1, linewidth=0, fmt=gcolors[obj])
		else:
			print 'Illegal measurement type'
			sys.exit(0)

	#fig.set_size_inches(20.,5.)
	#plt.savefig('fit.png',dpi=100,bbox_inches='tight')
	fig.set_size_inches(20.,5.)
	if args.dataset == 'sch':
		plt.savefig('fit.sch'+str(args.id)+label+'.pdf',dpi=100,bbox_inches='tight')
	elif args.dataset == 'yel':
		plt.savefig('fit.yel'+str(args.id)+label+'.pdf',dpi=100,bbox_inches='tight')
	elif args.dataset == 'lu':
		plt.savefig('fit.lu'+str(args.id)+label+'.pdf',dpi=100,bbox_inches='tight')
	elif args.dataset == 'do':
		plt.savefig('fit.do'+str(args.id)+label+'.pdf',dpi=100,bbox_inches='tight')
	elif args.dataset == 'unp':
		plt.savefig('fit.unp'+str(args.id)+label+'.pdf',dpi=100,bbox_inches='tight')
	else:
		plt.savefig('fit'+label+'.pdf',dpi=100,bbox_inches='tight')
	#plt.show()

sys.exit()
