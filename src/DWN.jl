module DWN
using LinearAlgebra, SpecialFunctions, FFTW

parsable(x::AbstractChar) = isdigit(x) || (x in ('.', ' ', '-'))

function mparse(s::AbstractString)
    flag = true
    for i in s
        flag &= parsable(i)
    end
    if flag
        if ' ' in s
            return parse.(Float64, split(strip(s), ' '; keepempty = false))
        else
            return parse(Float64, strip(s))
        end
    else
        return NaN
    end
end

"""
(model, dep, strike, dip, rake, slip, fault, rec, recdep,
npts, tl, tsource, t0, m, xl) = loadmodelinp(path)

load old style input file of dwn
"""
function loadmodelinp(path)
    l = readlines(path)
    nlayer = mparse(l[1]) |> Int
    model = Matrix{Float64}(undef, nlayer, 6)
    for il = 1:nlayer
        model[il, :] .= mparse(l[il+1])
    end
    (dep, strike, dip, rake, slip, fault) = mparse(join(l[nlayer+2:nlayer+5], ' '))
    (nrec, recdep) = mparse(l[nlayer+6])
    nrec = round(Int, nrec)
    rec = Tuple{Float64,Float64}[]
    for i = 1:nrec
        (dist, az) = mparse(l[nlayer+6+i])
        push!(rec, (dist, az))
    end
    (npts, tl, tsource, t0, m, xl) = mparse(join(l[nlayer+nrec+7:end], ' '))
    npts = round(Int, npts)
    m = round(Int, m)
    return (model, dep, strike, dip, rake, slip, fault, rec, recdep, npts, tl, tsource, t0, m, xl)
end

function matprod!(z::Matrix, x::Matrix, y::Matrix)
    @assert size(z, 1) == size(x, 1) && size(z, 2) == size(y, 2) && size(x, 2) == size(y, 1)
    for i = 1:size(z, 1), j = 1:size(z, 2)
        z[i, j] = zero(eltype(z))
        for k = 1:size(x, 2)
            z[i, j] += x[i, k] * y[k, j]
        end
    end
    return nothing
end

function matinv1m!(x::Matrix)
    o = one(eltype(x))
    d = (o - x[1, 1]) * (o - x[2, 2]) - x[2, 1] * x[1, 2]
    t = x[1, 1]
    x[1, 1] = (o - x[2, 2]) / d
    x[2, 2] = (o - t) / d
    x[1, 2] /= d
    x[2, 1] /= d
    return nothing
end

function preprocessLayer(model::NamedTuple, upperDepth::Real, lowerDepth::Real)
    topdep = cumsum(model.thickness)
    topdep[end] = Inf
    iUpper = findfirst(topdep .>= upperDepth)
    iLower = findfirst(topdep .>= lowerDepth)
    newDep = [topdep[1:(iUpper-1)]
              upperDepth
              topdep[iUpper:(iLower-1)]
              lowerDepth
              topdep[iLower:end]]
    newthick = [0.0; diff(newDep[1:(end-1)]); 0.0]
    newidx = [1:iUpper; iUpper:iLower; iLower:length(model.vp)]
    newmodel = (thickness = newthick, vp = model.vp[newidx], vs = model.vs[newidx], ρ = model.ρ[newidx],
                Qp = model.Qp[newidx], Qs = model.Qs[newidx])
    upperLayer = findfirst(newDep .== upperDepth)
    lowerLayer = findlast(newDep .== lowerDepth) + 1
    return (newmodel, upperLayer, lowerLayer)
end

"""
(cm11, cm22, cm33, cm12, cm13, cm23) = doublecouple2tensor(strike::Real, dip::Real, rake::Real)
"""
function doublecouple2tensor(strike::Real, dip::Real, rake::Real)
    cs = cosd(strike)
    ss = sind(strike)
    cdip = cosd(dip)
    sdip = sind(dip)
    cr = cosd(rake)
    sr = sind(rake)
    as1 = cr * cs + sr * cdip * ss
    as2 = cr * ss - sr * cdip * cs
    as3 = -sr * sdip
    an1 = -sdip * ss
    an2 = sdip * cs
    an3 = -cdip
    cm11 = 2 * as1 * an1
    cm22 = 2 * as2 * an2
    cm33 = 2 * as3 * an3
    cm12 = (as1 * an2 + as2 * an1)
    cm13 = (as1 * an3 + as3 * an1)
    cm23 = (as2 * an3 + as3 * an2)
    return (cm11, cm22, cm33, cm12, cm13, cm23)
end

function applysource!(du1, du2, du3, component, ampsv, amsh, k, ik, ik2, wza, wzb, sourceLayer, receiverLayer, ifreq,
                      su, sd, sdsh, sush, tup, tdw, quu, qud, qdu, qdd, ee, ntt, mt, tupsh, tdwsh, quush, qudsh, qdush,
                      qddsh, eesh, nttsh, mtsh, al, bl, alsh, blsh, psvcoef, shcoef)
    sd .= 0.0
    sdsh .= 0.0
    if component == 1
        c2 = ampsv[ifreq] * 2 * k^2
        c3 = ampsv[ifreq] * k * (k^2 / wzb[sourceLayer] - wzb[sourceLayer])
        c4 = ampsv[ifreq] * ik2 / 2
        sd[1, 4] = -c4 * k / wza[sourceLayer]
        sd[2, 4] = c4
        sd[1, 5] = ampsv[ifreq] * ik * (k^2 / wza[sourceLayer] / 2)
        sd[2, 5] = -ampsv[ifreq] * ik2 / 2
        sdsh[4] = -amsh[ifreq] * ik / (2 * wzb[sourceLayer])
    elseif component == 2
        c2 = ampsv[ifreq] * 2 * k^2
        c3 = ampsv[ifreq] * k * (k^2 / wzb[sourceLayer] - wzb[sourceLayer])
        c4 = -ampsv[ifreq] * ik2 / 2
        sd[1, 4] = -c4 * k / wza[sourceLayer]
        sd[2, 4] = c4
        sd[1, 5] = ampsv[ifreq] * ik * (k^2 / wza[sourceLayer] / 2)
        sd[2, 5] = -ampsv[ifreq] * ik2 / 2
        sdsh[4] = amsh[ifreq] * ik / (2 * wzb[sourceLayer])
    elseif component == 3
        sd[1, 5] = ampsv[ifreq] * ik * wza[sourceLayer]
        sd[2, 5] = ampsv[ifreq] * ik2
    elseif component == 4
        c1 = ampsv[ifreq] * ik2
        sd[1, 1] = -c1 * k / wza[sourceLayer]
        sd[2, 1] = c1
        sdsh[1] = amsh[ifreq] * ik / wzb[sourceLayer]
    elseif component == 5
        sd[1, 2] = ampsv[ifreq] * 2 * k^2
        sd[2, 2] = ampsv[ifreq] * k * (k^2 / wzb[sourceLayer] - wzb[sourceLayer])
        sdsh[2] = amsh[ifreq]
    elseif component == 6
        sd[1, 3] = ampsv[ifreq] * 2 * k^2
        sd[2, 3] = ampsv[ifreq] * k * (k^2 / wzb[sourceLayer] - wzb[sourceLayer])
        sdsh[3] = -amsh[ifreq]
    end
    @. su = sd * psvcoef
    @. sush = sdsh * shcoef
    if receiverLayer <= sourceLayer
        # bl .= tup * (quu[sourceLayer] * su + qud[sourceLayer] * ee[sourceLayer] * sd)
        # al .= ntt[receiverLayer] * bl
        matprod!(bl, ee[sourceLayer], sd)
        matprod!(al, qud[sourceLayer], bl)
        matprod!(bl, quu[sourceLayer], su)
        al .+= bl
        matprod!(bl, tup, al)
        matprod!(al, ntt[receiverLayer], bl)
        for i = 1:5
            blsh[i] = tupsh * (quush[sourceLayer] * sush[i] + qudsh[sourceLayer] * sdsh[i] * eesh[sourceLayer])
            alsh[i] = nttsh[receiverLayer] * blsh[i]
        end
    else
        # al .= tdw * (qdu[sourceLayer] * su + qdd[sourceLayer] * sd)
        # bl .= mt[receiverLayer] * al
        matprod!(al, qdu[sourceLayer], su)
        matprod!(bl, qdd[sourceLayer], sd)
        bl .+= al
        matprod!(al, tdw, bl)
        matprod!(bl, mt[receiverLayer], al)
        for i = 1:5
            alsh[i] = tdwsh * (qddsh[sourceLayer] * sdsh[i] + qdush[sourceLayer] * sush[i])
            blsh[i] = mtsh[receiverLayer] * alsh[i]
        end
    end
    for i = 1:5
        du1[i] = bl[1, i] + al[1, i] + wzb[receiverLayer] / k * (bl[2, i] - al[2, i])
        du2[i] = 1im * (wza[receiverLayer] * (bl[1, i] - al[1, i]) - k * (bl[2, i] + al[2, i]))
        du3[i] = blsh[i] + alsh[i]
    end
    return nothing
end

function caldej!(m, ej, rd, ru, td, tu, ee, mt, mb, nt, nb, ntt, quu, qud, qdu, qdd, f, g, tdw, tup, tdsh, tush, rdsh,
                 rush, eesh, mtsh, mbsh, nbsh, ntsh, nttsh, quush, qudsh, qdush, qddsh, fsh, sd, su, sdsh, sush, gsh,
                 al, bl, alsh, blsh, wza, wzb, woa, wob, du1, du2, du3, aj1r, ajkr, aj2, aj2r, aj1k, daj2, dej,
                 nreceiver, pil, wa2, wb2, ω, model, μ, cth, sourceLayer, receiverLayer, ampsv, amsh, sin_az, cos_az,
                 sin_2az, cos_2az, J0, J1, dist, ifreq, nmodellayer, tmpmat2, Imat, psvcoef, shcoef)
    k = pil * m
    ik = 1im * k
    ik2 = 1im * k^2
    for i = 1:nmodellayer
        wza[i] = sqrt(wa2[i, ifreq] - k^2)
        wzb[i] = sqrt(wb2[i, ifreq] - k^2)
        if imag(wza[i]) > 0
            wza[i] *= -1
        end
        if imag(wzb[i]) > 0
            wzb[i] *= -1
        end
    end
    @. woa = wza / ω[ifreq]
    @. wob = wzb / ω[ifreq]
    cu = k / ω[ifreq]
    for i = 1:(nmodellayer-1)
        cc = -2 * (μ[i+1, ifreq] - μ[i, ifreq])
        c1 = cc * cu^2
        c2 = c1 - model.ρ[i]
        c3 = c1 + model.ρ[i+1]
        c4 = c2 + c3 - c1
        c5 = c2^2
        c6 = c3^2
        c7 = c4^2 * cu^2
        a1 = model.ρ[i] * model.ρ[i+1]
        c8 = woa[i] * wob[i]
        c9 = woa[i] * wob[i+1]
        c10 = woa[i+1] * wob[i]
        c11 = woa[i+1] * wob[i+1]
        c14 = a1 * c9
        c15 = a1 * c10
        c16 = cc * c1 * c8 * c11
        c17 = c5 * c11
        c18 = c6 * c8
        d1d = c7 + c17 + c15
        d2d = c16 + c18 + c14
        d1u = c7 + c18 + c14
        d2u = c16 + c17 + c15
        c19 = c3 * wob[i] - c2 * wob[i+1]
        c20 = c3 * woa[i] - c2 * woa[i+1]
        dd = d1d + d2d
        c21 = 2 * cu * woa[i]
        c22 = 2 * cu * wob[i]
        c23 = 2 * cu * woa[i+1]
        c24 = 2 * cu * wob[i+1]
        c25 = (c4 * c3 + cc * c2 * c11) / dd
        c35 = (c4 * c2 + cc * c3 * c8) / dd
        c26 = model.ρ[i] / dd
        c27 = 2 * a1 * (c10 - c9)
        c36 = model.ρ[i+1] / dd
        rd[i][1, 1] = (d2d - d1d) / dd
        rd[i][1, 2] = c22 * c25
        rd[i][2, 1] = -c21 * c25
        rd[i][2, 2] = (d2d - d1d + c27) / dd
        ru[i][1, 1] = (d2u - d1u) / dd
        ru[i][1, 2] = -c24 * c35
        ru[i][2, 1] = c23 * c35
        ru[i][2, 2] = (d2u - d1u - c27) / dd
        td[i][1, 1] = 2 * c26 * woa[i] * c19
        td[i][1, 2] = c26 * c22 * (c4 + cc * c9)
        td[i][2, 1] = -c26 * c21 * (c4 + cc * c10)
        td[i][2, 2] = 2 * c26 * wob[i] * c20
        tu[i][1, 1] = 2 * c36 * woa[i+1] * c19
        tu[i][1, 2] = c36 * c24 * (c4 + cc * c10)
        tu[i][2, 1] = -c36 * c23 * (c4 + cc * c9)
        tu[i][2, 2] = 2 * c36 * wob[i+1] * c20
    end
    for i = 1:nmodellayer
        ee[i][1, 1] = exp(cth[i] * wza[i])
        ee[i][2, 2] = exp(cth[i] * wzb[i])
        ee[i][1, 2] = 0.0
        ee[i][2, 1] = 0.0
    end
    ee[1] .= 0.0 + 0.0im
    mb[end-1] .= rd[end-1]
    for l = (nmodellayer-1):-1:2
        # mt[l] .= ee[l] * mb[l] * ee[l]
        # mb[l-1] .= rd[l-1] + tu[l-1] / (I - mt[l] * ru[l-1]) * mt[l] * td[l-1]
        matprod!(tmpmat2, ee[l], mb[l])
        matprod!(mt[l], tmpmat2, ee[l])
        matprod!(tmpmat2, mt[l], ru[l-1])
        matinv1m!(tmpmat2)
        matprod!(mb[l-1], tu[l-1], tmpmat2)
        matprod!(tmpmat2, mb[l-1], mt[l])
        matprod!(mb[l-1], tmpmat2, td[l-1])
        mb[l-1] .+= rd[l-1]
    end

    nt[1] .= ru[1]
    for l = 2:nmodellayer
        # nb[l] .= ee[l] * nt[l-1] * ee[l]
        # nt[l] .= ru[l] + td[l] / (I - nb[l] * rd[l]) * nb[l] * tu[l]
        matprod!(tmpmat2, ee[l], nt[l-1])
        matprod!(nb[l], tmpmat2, ee[l])
        matprod!(tmpmat2, nb[l], rd[l])
        matinv1m!(tmpmat2)
        matprod!(nt[l], td[l], tmpmat2)
        matprod!(tmpmat2, nt[l], nb[l])
        matprod!(nt[l], tmpmat2, tu[l])
        nt[l] .+= ru[l]
    end

    for l = 2:nmodellayer
        # ntt[l] .= ee[l] * nt[l-1] * ee[l]
        # quu[l] .= inv(I - mt[l] * nt[l-1])
        # qud[l] .= (I - mt[l] * nt[l-1]) \ ee[l] * mb[l]
        # qdu[l] .= (I - nb[l] * mb[l]) \ nb[l]
        # qdd[l] .= inv(I - nb[l] * mb[l])
        matprod!(tmpmat2, ee[l], nt[l-1])
        matprod!(ntt[l], tmpmat2, ee[l])

        matprod!(quu[l], mt[l], nt[l-1])
        matinv1m!(quu[l])

        matprod!(tmpmat2, quu[l], ee[l])
        matprod!(qud[l], tmpmat2, mb[l])

        matprod!(qdd[l], nb[l], mb[l])
        matinv1m!(qdd[l])

        matprod!(qdu[l], qdd[l], nb[l])
    end
    qud[1] .= mb[1]

    for l = 1:(nmodellayer-1)
        # f[l] .= (I(2) - ru[l] * mt[l+1]) \ td[l]
        # g[l] .= (I(2) - rd[l] * nb[l]) \ tu[l]
        matprod!(tmpmat2, ru[l], mt[l+1])
        matinv1m!(tmpmat2)
        matprod!(f[l], tmpmat2, td[l])
        matprod!(tmpmat2, rd[l], nb[l])
        matinv1m!(tmpmat2)
        matprod!(g[l], tmpmat2, tu[l])
    end
    f[nmodellayer] .= td[nmodellayer-1]

    @. tdw = Imat
    @. tup = Imat
    # for i = 1:2
    #     tdw[i, i] = 1.0 + 0.0im
    #     tdw[i, 3-i] = 0.0 + 0.0im
    #     tdw[3-i, i] = 0.0 + 0.0im
    #     tup[i, i] = 1.0 + 0.0im
    #     tup[i, 3-i] = 0.0 + 0.0im
    #     tup[3-i, i] = 0.0 + 0.0im
    # end
    for l = (receiverLayer-1):-1:sourceLayer
        # tdw *= f[l]
        matprod!(tmpmat2, tdw, f[l])
        tdw .= tmpmat2
        if l == sourceLayer
            continue
        end
        # tdw *= ee[l]
        matprod!(tmpmat2, tdw, ee[l])
        tdw .= tmpmat2
    end

    for l = receiverLayer:(sourceLayer-1)
        if l != receiverLayer
            # tup *= ee[l]
            matprod!(tmpmat2, tup, ee[l])
            tup .= tmpmat2
        end
        # tup *= g[l]
        matprod!(tmpmat2, tup, g[l])
        tup .= tmpmat2
    end

    # c1 = μ[:, ifreq] .* wob
    # c2 = [c1[2:end]; 0.0]
    # @. eesh = exp(cth * wzb)
    # @. rdsh = (c1 - c2) / (c1 + c2)
    # rdsh[end] = 0.0
    # @. tdsh = 2.0 * c1 / (c1 + c2)
    # tdsh[end] = 0.0
    # @. rush = -rdsh
    # @. tush = 2.0 * c2 / (c1 + c2)
    # tush[end] = 0.0
    @views for l = 1:nmodellayer-1
        c1 = μ[l, ifreq] * wob[l]
        c2 = μ[l+1, ifreq] * wob[l+1]
        eesh[l] = exp(cth[l] * wzb[l])
        rdsh[l] = (c1 - c2) / (c1 + c2)
        tdsh[l] = 2.0 * c1 / (c1 + c2)
        rush[l] = -rdsh[l]
        tush[l] = 2.0 * c2 / (c1 + c2)
    end
    rdsh[end] = 0.0
    tdsh[end] = 0.0
    rush[end] = 0.0
    tush[end] = 0.0

    mbsh[end-1] = rdsh[end-1]
    ntsh[1] = rush[1]
    fsh[end-1] = tdsh[end-1]
    gsh[1] = tush[1]
    for l = (nmodellayer-1):-1:2
        mtsh[l] = mbsh[l] * eesh[l]^2
        mbsh[l-1] = rdsh[l-1] + tdsh[l-1] * tush[l-1] * mtsh[l] / (1 - rush[l-1] * mtsh[l])
    end
    for l = 1:(nmodellayer-2)
        nbsh[l] = ntsh[l] * eesh[l+1]^2
        ntsh[l+1] = rush[l+1] + tdsh[l+1] * tush[l+1] * nbsh[l] / (1 - rdsh[l+1] * nbsh[l])
    end
    qudsh[1] = mbsh[1]
    for i = 2:nmodellayer
        nttsh[i] = eesh[i]^2 * ntsh[i-1]
        quush[i] = 1 / (1 - mtsh[i] * ntsh[i-1])
        qudsh[i] = eesh[i] * mbsh[i] / (1 - mtsh[i] * ntsh[i-1])
        qdush[i] = nbsh[i-1] / (1 - nbsh[i-1] * mbsh[i])
        qddsh[i] = 1 / (1 - nbsh[i-1] * mbsh[i])
    end
    qddsh[end] = 0
    # @. fsh[1:(end-2)] = tdsh[2:(end-1)] / (1 - rush[2:(end-1)] * mtsh[2:(end-1)])
    # @. gsh[2:end] = tush[2:end] / (1 - rdsh[2:end] * nbsh[1:(end-1)])
    for l = 2:nmodellayer-1
        fsh[l-1] = tdsh[l] / (1 - rush[l] * mtsh[l])
        gsh[l] = tush[l] / (1 - rdsh[l] * nbsh[l-1])
    end
    gsh[end] = tush[end] / (1 - rdsh[end] * nbsh[end-1])

    tdwsh = 1
    tupsh = 1
    for l = (receiverLayer-1):-1:sourceLayer
        if l != sourceLayer
            tdwsh *= eesh[l]
        end
        tdwsh *= fsh[l-1]
    end
    for l = receiverLayer:(sourceLayer-1)
        if l != receiverLayer
            tupsh *= eesh[l]
        end
        tupsh *= gsh[l]
    end
    @views for mc = 1:6
        applysource!(du1, du2, du3, mc, ampsv, amsh, k, ik, ik2, wza, wzb, sourceLayer, receiverLayer, ifreq, su, sd,
                     sdsh, sush, tup, tdw, quu, qud, qdu, qdd, ee, ntt, mt, tupsh, tdwsh, quush, qudsh, qdush, qddsh,
                     eesh, nttsh, mtsh, al, bl, alsh, blsh, psvcoef, shcoef)
        for i = 1:nreceiver
            aj1r[i] = J1[i, m] / dist[i]
            ajkr[i] = k * J0[i, m] - aj1r[i]
            aj2[i] = 2 * aj1r[i] / k - J0[i, m]
            aj2r[i] = 2 * aj2[i] / dist[i]
            aj1k[i] = k * J1[i, m]
            daj2[i] = aj1k[i] - aj2r[i]
            dej[i, 1, mc] = sin_2az[i] * (daj2[i] * du1[1] - aj2r[i] * du3[1]) +
                            cos_az[i] * (ajkr[i] * du1[2] + aj1r[i] * du3[2]) +
                            sin_az[i] * (ajkr[i] * du1[3] - aj1r[i] * du3[3]) +
                            cos_2az[i] * (daj2[i] * du1[4] + aj2r[i] * du3[4]) - aj1k[i] * du1[5]
            dej[i, 2, mc] = cos_2az[i] * (aj2r[i] * du1[1] - daj2[i] * du3[1]) -
                            sin_az[i] * (aj1r[i] * du1[2] + ajkr[i] * du3[2]) +
                            cos_az[i] * (aj1r[i] * du1[3] - ajkr[i] * du3[3]) -
                            sin_2az[i] * (aj2r[i] * du1[4] + daj2[i] * du3[4])
            dej[i, 3, mc] = aj2[i] * (sin_2az[i] * du2[1] + cos_2az[i] * du2[4]) +
                            J1[i, m] * (cos_az[i] * du2[2] + sin_az[i] * du2[3]) +
                            J0[i, m] * du2[5]
            for j = 1:3
                ej[i, j, mc] += dej[i, j, mc]
            end
        end
    end
    return nothing
end

"""
dwn(inputModel, hypocentralDepth, faultLength, receiver, receiverDepth, npts, dt,
periodicityLength, tsource, maxSeriesOrder=10000)

calculate Green function of each station

    inputModel Nx6 matrix. format
    `thickness,vp,vs,rho,Qp,Qs`
    hypocentralDepth source depth
    faultLength
    receiver list of (distance, azimuth). `azimuth` is in degree
    receiverDepth
    npts
    dt
    maxSeriesOrder default 10000
    periodicityLength
    tsource
"""
function dwn(inputModel::AbstractMatrix, hypocentralDepth::Real,faultLength::Real, receiver::Vector{<:Tuple},
    receiverDepth::Real, npts::Integer, dt::Real, periodicityLength::Real, tsource::Real, maxSeriesOrder::Integer = 10000)
    timeLength = npts * dt
    nreceiver = length(receiver)
    model = let
        modelAirlayer = [0.0 0.34 0.001 1.3e-3 1.0e6 1.0e6; inputModel]
        (thickness = modelAirlayer[:, 1], vp = modelAirlayer[:, 2], vs = modelAirlayer[:, 3], ρ = modelAirlayer[:, 4],
         Qp = modelAirlayer[:, 5], Qs = modelAirlayer[:, 6])
    end

    if hypocentralDepth > receiverDepth
        (model, receiverLayer, sourceLayer) = preprocessLayer(model, receiverDepth, hypocentralDepth)
    else
        (model, sourceLayer, receiverLayer) = preprocessLayer(model, hypocentralDepth, receiverDepth)
    end

    nmodellayer = length(model.thickness)
    dist = zeros(nreceiver)
    az = zeros(nreceiver)
    for i = 1:nreceiver
        dist[i] = receiver[i][1]
        az[i] = receiver[i][2]
    end

    cos_az = cosd.(az)
    sin_az = sind.(az)
    cos_2az = @. cosd(2 * az)
    sin_2az = @. sind(2 * az)

    q = timeLength / 2
    pil = 2 * π / periodicityLength
    dfreq = 1.0 / timeLength
    nfreq = round(Int, npts / 2)
    aw = -π / q

    aqp = zeros(ComplexF64, nmodellayer)
    aqs = zeros(ComplexF64, nmodellayer)
    cth = zeros(ComplexF64, nmodellayer)
    @. aqp = (1 + 1im / (2 * model.Qp)) / (1 + 0.25 / model.Qp^2) * model.vp
    @. aqs = (1 + 1im / (2 * model.Qs)) / (1 + 0.25 / model.Qs^2) * model.vs
    @. cth = -1im * model.thickness

    J0 = zeros(nreceiver, maxSeriesOrder)
    J1 = zeros(nreceiver, maxSeriesOrder)
    for i = 1:nreceiver, j = 1:maxSeriesOrder
        J0[i, j] = besselj0(2 * π * j * dist[i] / periodicityLength)
        J1[i, j] = besselj1(2 * π * j * dist[i] / periodicityLength)
    end
    ω = zeros(ComplexF64, nfreq)
    zom = zeros(ComplexF64, nfreq)
    ϕ = zeros(ComplexF64, nfreq)
    xlnf = zeros(ComplexF64, nfreq)
    a0 = zeros(ComplexF64, nfreq)
    α = zeros(ComplexF64, nfreq)
    β = zeros(ComplexF64, nfreq)
    μ = zeros(ComplexF64, nmodellayer, nfreq)
    wa2 = zeros(ComplexF64, nmodellayer, nfreq)
    wb2 = zeros(ComplexF64, nmodellayer, nfreq)
    ampsv = zeros(ComplexF64, nfreq)
    amsh = zeros(ComplexF64, nfreq)
    c_a0 = dfreq * faultLength^2 / (2 * periodicityLength * model.ρ[sourceLayer])
    @. ω = (0:nfreq-1) * dfreq * 2.0 * π + 1im * aw
    @. zom = abs(ω) / (2 * π)
    @. ϕ = angle(ω)
    @. xlnf = (1im * ϕ + log(zom) - log(1 / tsource)) / π
    for ilayer = 1:nmodellayer
        @. α = aqp[ilayer] / (1 - xlnf / model.Qp[ilayer])
        @. β = aqs[ilayer] / (1 - xlnf / model.Qs[ilayer])
        α[1] = model.vp[ilayer]
        β[1] = model.vs[ilayer]
        @. μ[ilayer, :] = β^2 * model.ρ[ilayer]
        @. wa2[ilayer, :] = (ω / α)^2
        @. wb2[ilayer, :] = (ω / β)^2
    end
    @. a0 = μ[sourceLayer, :] * c_a0
    @. ampsv = a0 / ω^2
    @. amsh = a0 / (aqs[sourceLayer] / (1 - xlnf / model.Qs[sourceLayer]))^2

    rd = Vector{Matrix{ComplexF64}}(undef, nmodellayer)
    ru = Vector{Matrix{ComplexF64}}(undef, nmodellayer)
    td = Vector{Matrix{ComplexF64}}(undef, nmodellayer)
    tu = Vector{Matrix{ComplexF64}}(undef, nmodellayer)
    ee = Vector{Matrix{ComplexF64}}(undef, nmodellayer)
    mt = Vector{Matrix{ComplexF64}}(undef, nmodellayer)
    mb = Vector{Matrix{ComplexF64}}(undef, nmodellayer)
    nt = Vector{Matrix{ComplexF64}}(undef, nmodellayer)
    nb = Vector{Matrix{ComplexF64}}(undef, nmodellayer)
    ntt = Vector{Matrix{ComplexF64}}(undef, nmodellayer)
    quu = Vector{Matrix{ComplexF64}}(undef, nmodellayer)
    qud = Vector{Matrix{ComplexF64}}(undef, nmodellayer)
    qdu = Vector{Matrix{ComplexF64}}(undef, nmodellayer)
    qdd = Vector{Matrix{ComplexF64}}(undef, nmodellayer)
    f = Vector{Matrix{ComplexF64}}(undef, nmodellayer)
    g = Vector{Matrix{ComplexF64}}(undef, nmodellayer)
    for i = 1:nmodellayer
        rd[i] = zeros(ComplexF64, 2, 2)
        ru[i] = zeros(ComplexF64, 2, 2)
        td[i] = zeros(ComplexF64, 2, 2)
        tu[i] = zeros(ComplexF64, 2, 2)
        ee[i] = zeros(ComplexF64, 2, 2)
        mt[i] = zeros(ComplexF64, 2, 2)
        mb[i] = zeros(ComplexF64, 2, 2)
        nt[i] = zeros(ComplexF64, 2, 2)
        nb[i] = zeros(ComplexF64, 2, 2)
        ntt[i] = zeros(ComplexF64, 2, 2)
        quu[i] = zeros(ComplexF64, 2, 2)
        qud[i] = zeros(ComplexF64, 2, 2)
        qdu[i] = zeros(ComplexF64, 2, 2)
        qdd[i] = zeros(ComplexF64, 2, 2)
        f[i] = zeros(ComplexF64, 2, 2)
        g[i] = zeros(ComplexF64, 2, 2)
    end
    tdw = zeros(ComplexF64, 2, 2)
    tup = zeros(ComplexF64, 2, 2)
    tdsh = zeros(ComplexF64, nmodellayer)
    tush = zeros(ComplexF64, nmodellayer)
    rdsh = zeros(ComplexF64, nmodellayer)
    rush = zeros(ComplexF64, nmodellayer)
    eesh = zeros(ComplexF64, nmodellayer)
    mtsh = zeros(ComplexF64, nmodellayer)
    mbsh = zeros(ComplexF64, nmodellayer)
    nbsh = zeros(ComplexF64, nmodellayer)
    ntsh = zeros(ComplexF64, nmodellayer)
    nttsh = zeros(ComplexF64, nmodellayer)
    quush = zeros(ComplexF64, nmodellayer)
    qudsh = zeros(ComplexF64, nmodellayer)
    qdush = zeros(ComplexF64, nmodellayer)
    qddsh = zeros(ComplexF64, nmodellayer)
    fsh = zeros(ComplexF64, nmodellayer)
    sd = zeros(ComplexF64, 2, 5)
    su = zeros(ComplexF64, 2, 5)
    sdsh = zeros(ComplexF64, 1, 5)
    sush = zeros(ComplexF64, 1, 5)
    gsh = zeros(ComplexF64, nmodellayer)
    wza = zeros(ComplexF64, nmodellayer)
    wzb = zeros(ComplexF64, nmodellayer)
    woa = zeros(ComplexF64, nmodellayer)
    wob = zeros(ComplexF64, nmodellayer)
    tmpmat2 = zeros(ComplexF64, 2, 2)
    Imat = zeros(ComplexF64, 2, 2)
    du1 = zeros(ComplexF64, 5)
    du2 = zeros(ComplexF64, 5)
    du3 = zeros(ComplexF64, 5)
    aj1r = zeros(Float64, nreceiver)
    ajkr = zeros(Float64, nreceiver)
    aj2 = zeros(Float64, nreceiver)
    aj2r = zeros(Float64, nreceiver)
    aj1k = zeros(Float64, nreceiver)
    daj2 = zeros(Float64, nreceiver)
    dej = zeros(ComplexF64, nreceiver, 3, 6)
    ej = zeros(ComplexF64, nreceiver, 3, 6)
    al = zeros(ComplexF64, 2, 5)
    bl = zeros(ComplexF64, 2, 5)
    alsh = zeros(ComplexF64, 2, 5)
    blsh = zeros(ComplexF64, 2, 5)
    psvcoef = [1.0 -1.0 -1.0 1.0 1.0
               -1.0 1.0 1.0 -1.0 -1.0]
    shcoef = [1.0 -1.0 -1.0 1.0 0.0]
    Imat[1, 1] = 1.0
    Imat[2, 2] = 1.0

    u = Array{Vector{ComplexF64},3}(undef, nreceiver, 3, 6)
    for i = 1:nreceiver, j = 1:3, k = 1:6
        u[i, j, k] = zeros(ComplexF64, nfreq)
    end
    for ifreq = 1:nfreq
        ej .= 0.0 + 0.0im
        m = 1
        while m <= maxSeriesOrder
            caldej!(m, ej, rd, ru, td, tu, ee, mt, mb, nt, nb, ntt, quu, qud, qdu, qdd, f, g, tdw, tup, tdsh, tush,
                    rdsh, rush, eesh, mtsh, mbsh, nbsh, ntsh, nttsh, quush, qudsh, qdush, qddsh, fsh, sd, su, sdsh,
                    sush, gsh, al, bl, alsh, blsh, wza, wzb, woa, wob, du1, du2, du3, aj1r, ajkr, aj2, aj2r, aj1k,
                    daj2, dej, nreceiver, pil, wa2, wb2, ω, model, μ, cth, sourceLayer, receiverLayer, ampsv, amsh,
                    sin_az, cos_az, sin_2az, cos_2az, J0, J1, dist, ifreq, nmodellayer, tmpmat2, Imat, psvcoef, shcoef)
            flag = true
            for i = 1:nreceiver, j = 1:3, k = 1:6
                flag &= abs(dej[i, j, k]) <= 1e-4 * abs(ej[i, j, k])
            end
            if flag
                break
            end
            m += 1
        end
        if m > maxSeriesOrder
            @warn "maxSeriesOrder triggered"
        end
        # println(ifreq, " ", m)
        for i = 1:nreceiver, j = 1:6
            u[i, 1, j][ifreq] = ej[i, 1, j] * cos_az[i] - ej[i, 2, j] * sin_az[i]
            u[i, 2, j][ifreq] = ej[i, 2, j] * cos_az[i] + ej[i, 1, j] * sin_az[i]
            u[i, 3, j][ifreq] = ej[i, 3, j]
        end
    end
    return u
end

"""
dwn(inputModel, hypocentralDepth, faultLength, receiver, receiverDepth, npts, dt, maxSeriesOrder=10000)

calculate Green function of each station

    inputModel Nx6 matrix. format
    `thickness,vp,vs,rho,Qp,Qs`
    hypocentralDepth source depth
    faultLength
    receiver list of (distance, azimuth). `azimuth` is in degree
    receiverDepth
    npts
    dt
    maxSeriesOrder default 10000
"""
function dwn(inputModel, hypocentralDepth, faultLength, receiver, receiverDepth, npts, dt, maxSeriesOrder::Int = 10000)
    xl = maximum(inputModel[:, 2]) * npts * dt + maximum(map(v->v[1], receiver))
    return dwn(inputModel, hypocentralDepth, faultLength, receiver, receiverDepth, npts, dt, xl, dt, maxSeriesOrder)
end

"""
dwn(inputModel, hypocentralDepth, faultLength, receiver, receiverDepth, tl, dt, maxSeriesOrder=10000)

calculate Green function of each station

    inputModel Nx6 matrix. format
    `thickness,vp,vs,rho,Qp,Qs`
    hypocentralDepth source depth
    faultLength
    receiver list of (distance, azimuth). `azimuth` is in degree
    receiverDepth
    tl
    dt
    maxSeriesOrder default 10000
"""
function dwn(inputModel, hypocentralDepth, faultLength, receiver, receiverDepth, tl, dt, maxSeriesOrder::Int = 10000)
    npts = round(Int, tl/dt)
    npts += mod(npts, 2)
    xl = maximum(inputModel[:, 2]) * npts * dt + maximum(map(v->v[1], receiver))
    return dwn(inputModel, hypocentralDepth, faultLength, receiver, receiverDepth, npts, dt, xl, dt, maxSeriesOrder)
end

function freqspec2timeseries!(w::Vector, spec::Vector, amp::Vector, t::Vector, stf::Vector)
    t[1] = spec[1] * stf[1]
    for i = 2:length(spec)
        t[i] = spec[i] * stf[i]
        t[end-i+2] = conj(spec[i] * stf[i])
    end
    ifft!(t)
    @. w = real(t) * amp
    return nothing
end

"""
w = freqspec2timeseries(u, npts=0; stf = ComplexF64[])
"""
function freqspec2timeseries(u::Array{Vector{T}}, npts::Int = 0;
                             stf::Vector{S} = ComplexF64[]) where {T<:Complex,S<:Complex}
    if npts == 0
        npts = 2 * length(u[1])
    end
    if isempty(stf)
        stf = ones(ComplexF64, length(u[1]))
    end
    amp = @. exp(2 * π * (0:npts-1) / npts) * npts
    t = zeros(ComplexF64, npts)
    w = Array{Vector{Float64},3}(undef, size(u))
    for i in eachindex(u)
        w[i] = zeros(npts)
        freqspec2timeseries!(w[i], u[i], amp, t, stf)
    end
    return w
end

"""
    sourcetimefunction_v(npts::Int, nfreq::Int, timelength::Real, tsource::Real, t0::Real, slip::Real) -> (v, source_v)
"""
function sourcetimefunction_v(npts::Int, nfreq::Int, timelength::Real, tsource::Real, t0::Real, slip::Real)
    ω = zeros(ComplexF64, nfreq)
    source_v = zeros(ComplexF64, nfreq)
    @. ω = ((0:nfreq-1) - 1im) / timelength * 2.0 * π
    @. source_v = π * tsource / 4 / sinh(π * ω * tsource / 4) * exp(1im * ω * t0) * slip * ω
    v = zeros(npts)
    t = zeros(ComplexF64, npts)
    freqspec2timeseries!(v, source_v, fill(1.0, npts), t, fill(1.0 + 0.0im, nfreq))
    return (v, source_v)
end

export dwn, freqspec2timeseries, loadmodelinp, sourcetimefunction_v
end
