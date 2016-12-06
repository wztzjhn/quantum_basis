      subroutine arpack_zndrv1(n,nev,ncv,eigenvals,eigenvecs,lanczos_error)
      use sparse_cmplx
      use globals, only : hamiltonian_lil,hamiltonian_csr
      implicit none

      integer :: n, nev, ncv
      double precision :: eigenvals(nev), lanczos_error(nev)
      complex*16 :: eigenvecs(dim*nev)

      integer           maxn, maxnev, maxncv, ldv
      parameter         (maxn=2000000,maxnev=20,maxncv=60,ldv=maxn)
c
      integer           iparam(11), ipntr(14)
      logical           select(maxncv)
      Complex*16
     &                  ax(maxn), d(maxncv),
     &                  v(ldv,maxncv), workd(3*maxn),
     &                  workev(3*maxncv), resid(maxn),
     &                  workl(3*maxncv*maxncv+5*maxncv)
      Double precision
     &                  rwork(maxncv), rd(maxncv,3)
c
      character         bmat*1, which*2
      integer           ido, lworkl, info, j,jmin,
     &                  ierr, nconv, maxitr, ishfts, mode
      Complex*16        sigma
      Double precision  tol
      logical           rvec
c
c     %-----------------------------%
c     | BLAS & LAPACK routines used |
c     %-----------------------------%
c
      Double precision
     &                  dznrm2
      external          dznrm2 , zaxpy
c
c     %-----------------------%
c     | Executable Statements |
c     %-----------------------%
c
c     %--------------------------------------------------%
c     |                   N <= MAXN                      |
c     |                 NEV <= MAXNEV                    |
c     |           NEV + 2 <= NCV <= MAXNCV               |
c     %--------------------------------------------------%
c
      n   = dim
      nev = nev_in
      ncv = ncv_in
c      eigenvals=huge(1.d20)
c      lanczos_error=huge(1.d20)
c      jmin=-1
c      call lil_to_csr(hamiltonian_lil,hamiltonian_csr)
c      call lil_destroy(hamiltonian_lil)
c      if(hamiltonian_csr%sym/='S') stop "Hamiltonian not Hermitian!"

c      n=hamiltonian_csr%n_col
      if ( n .gt. maxn ) then
         print *, ' ERROR with _NDRV1: N is greater than MAXN '
         go to 9000
      else if ( nev .gt. maxnev ) then
         print *, ' ERROR with _NDRV1: NEV is greater than MAXNEV '
         go to 9000
      else if ( ncv .gt. maxncv ) then
         print *, ' ERROR with _NDRV1: NCV is greater than MAXNCV '
         go to 9000
      end if
      bmat  = 'I'
      which = 'SR'
c
c     %---------------------------------------------------%
c     | The work array WORKL is used in ZNAUPD  as         |
c     | workspace.  Its dimension LWORKL is set as        |
c     | illustrated below.  The parameter TOL determines  |
c     | the stopping criterion. If TOL<=0, machine        |
c     | precision is used.  The variable IDO is used for  |
c     | reverse communication, and is initially set to 0. |
c     | Setting INFO=0 indicates that a random vector is  |
c     | generated to start the ARNOLDI iteration.         |
c     %---------------------------------------------------%
c
      lworkl  = 3*ncv**2+5*ncv
      tol    = 0.0d0
      ido    = 0
      info   = 0
c
c     %---------------------------------------------------%
c     | This program uses exact shift with respect to     |
c     | the current Hessenberg matrix (IPARAM(1) = 1).    |
c     | IPARAM(3) specifies the maximum number of Arnoldi |
c     | iterations allowed.  Mode 1 of ZNAUPD  is used     |
c     | (IPARAM(7) = 1). All these options can be changed |
c     | by the user. For details see the documentation in |
c     | ZNAUPD .                                           |
c     %---------------------------------------------------%
c
      ishfts = 1
      maxitr = 5000
      mode   = 1
c
      iparam(1) = ishfts
      iparam(3) = maxitr
      iparam(7) = mode
c
c     %-------------------------------------------%
c     | M A I N   L O O P (Reverse communication) |
c     %-------------------------------------------%
c
 10   continue
c
c        %---------------------------------------------%
c        | Repeatedly call the routine ZNAUPD  and take |
c        | actions indicated by parameter IDO until    |
c        | either convergence is indicated or maxitr   |
c        | has been exceeded.                          |
c        %---------------------------------------------%
c
         call znaupd  ( ido, bmat, n, which, nev, tol, resid, ncv,
     &        v, ldv, iparam, ipntr, workd, workl, lworkl,
     &        rwork,info )
c
         if (ido .eq. -1 .or. ido .eq. 1) then
c
c           %-------------------------------------------%
c           | Perform matrix vector multiplication      |
c           |                y <--- OP*x                |
c           | The user should supply his/her own        |
c           | matrix vector multiplication routine here |
c           | that takes workd(ipntr(1)) as the input   |
c           | vector, and return the matrix vector      |
c           | product to workd(ipntr(2)).               |
c           %-------------------------------------------%
c
         call csr_times_vec(hamiltonian_csr,
     &                         workd(ipntr(1)), workd(ipntr(2)))
c
c           %-----------------------------------------%
c           | L O O P   B A C K to call ZNAUPD  again. |
c           %-----------------------------------------%
c
            go to 10

         end if
c
c     %----------------------------------------%
c     | Either we have convergence or there is |
c     | an error.                              |
c     %----------------------------------------%
c
      if ( info .lt. 0 ) then
c
c        %--------------------------%
c        | Error message, check the |
c        | documentation in ZNAUPD   |
c        %--------------------------%
c
         print *, ' '
         print *, ' Error with _naupd, info = ', info
         print *, ' Check the documentation of _naupd'
         print *, ' '
c
      else
c
c        %-------------------------------------------%
c        | No fatal errors occurred.                 |
c        | Post-Process using ZNEUPD .                |
c        |                                           |
c        | Computed eigenvalues may be extracted.    |
c        |                                           |
c        | Eigenvectors may also be computed now if  |
c        | desired.  (indicated by rvec = .true.)    |
c        %-------------------------------------------%
c
         rvec = .true.
c
         call zneupd  (rvec, 'A', select, d, v, ldv, sigma,
     &        workev, bmat, n, which, nev, tol, resid, ncv,
     &        v, ldv, iparam, ipntr, workd, workl, lworkl,
     &        rwork, ierr)
c
c        %----------------------------------------------%
c        | Eigenvalues are returned in the one          |
c        | dimensional array D.  The corresponding      |
c        | eigenvectors are returned in the first NCONV |
c        | (=IPARAM(5)) columns of the two dimensional  |
c        | array V if requested.  Otherwise, an         |
c        | orthogonal basis for the invariant subspace  |
c        | corresponding to the eigenvalues in D is     |
c        | returned in V.                               |
c        %----------------------------------------------%
c
         if ( ierr .ne. 0) then
c
c           %------------------------------------%
c           | Error condition:                   |
c           | Check the documentation of ZNEUPD . |
c           %------------------------------------%
c
             print *, ' '
             print *, ' Error with zneupd, info = ', ierr
             print *, ' Check the documentation of _neupd. '
             print *, ' '
c
         else
c
             nconv = iparam(5)
             do 20 j=1, nconv
c
c               %---------------------------%
c               | Compute the residual norm |
c               |                           |
c               |   ||  A*x - lambda*x ||   |
c               |                           |
c               | for the NCONV accurately  |
c               | computed eigenvalues and  |
c               | eigenvectors.  (iparam(5) |
c               | indicates how many are    |
c               | accurate to the requested |
c               | tolerance)                |
c               %---------------------------%
c
                call csr_times_vec(hamiltonian_csr, v(1,j), ax)
                call zaxpy (n, -d(j), v(1,j), 1, ax, 1)
                rd(j,1) = dble (d(j))
                rd(j,2) = dimag (d(j))
                rd(j,3) = dznrm2 (n, ax, 1)
!                rd(j,3) = rd(j,3) / dlapy2 (rd(j,1),rd(j,2))
                if(rd(j,1)<eigenvals) then
                    jmin=j
                    eigenvals=rd(j,1)
                    lanczos_error=rd(j,3)
                endif
 20          continue
             if(jmin/=-1) then
                 eigenvecs(1:n)=v(1:n,jmin)
             else
                 stop "gournd state wave function NOT found!"
             endif
c            %-----------------------------%
c            | Display computed residuals. |
c            %-----------------------------%
c
             call dmout (6, nconv, 3, rd, maxncv, -6,
     &            'Ritz values (Real, Imag) and absolute residuals')
          end if
c
c        %-------------------------------------------%
c        | Print additional convergence information. |
c        %-------------------------------------------%
c
         if ( info .eq. 1) then
             print *, ' '
             print *, ' Maximum number of iterations reached.'
             print *, ' '
         else if ( info .eq. 3) then
             print *, ' '
             print *, ' No shifts could be applied during implicit',
     &                ' Arnoldi update, try increasing NCV.'
             print *, ' '
         end if
c
         print *, ' '
         print *, 'ZNDRV1'
         print *, '====== '
         print *, ' '
         print *, ' Size of the matrix is ', n
         print *, ' The number of Ritz values requested is ', nev
         print *, ' The number of Arnoldi vectors generated',
     &            ' (NCV) is ', ncv
         print *, ' What portion of the spectrum: ', which
         print *, ' The number of converged Ritz values is ',
     &              nconv
         print *, ' The number of Implicit Arnoldi update',
     &            ' iterations taken is ', iparam(3)
         print *, ' The number of OP*x is ', iparam(9)
         print *, ' The convergence criterion is ', tol
         print *, ' '
c
      end if
c
c     %---------------------------%
c     | Done with program zndrv1 . |
c     %---------------------------%
c
 9000 continue
      end
