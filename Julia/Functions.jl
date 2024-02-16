using StaticArrays
using LinearAlgebra

# Bloch Functions
function rotation_b1scale(M0, gx, gy, resol, Xpos, Ypos, w1, phiRF, off_res, scale)
    @inbounds @views begin
        γ = 2 * π * 4258.0
        Mxyz_init1 = M0[1]
        Mxyz_init2 = M0[2]
        Mxyz_init3 = M0[3]

        Weff1 = scale * w1 * cos(phiRF)
        Weff2 = scale * w1 * -sin(phiRF)
        Weff3 = γ * (gx * Xpos + gy * Ypos) + off_res
        
        abs_weff = sqrt(Weff1^2 + Weff2^2 + Weff3^2)
        phi = -abs_weff * resol
        
        if abs_weff != 0.0
            Weff1 = Weff1 / abs_weff
            Weff2 = Weff2 / abs_weff
            Weff3 = Weff3 / abs_weff
        else
            Weff1 = 0.0
            Weff2 = 0.0
            Weff3 = 0.0
        end

        crs_prd1 = Weff2 * Mxyz_init3 - Weff3 * Mxyz_init2
        crs_prd2 = Weff3 * Mxyz_init1 - Weff1 * Mxyz_init3
        crs_prd3 = Weff1 * Mxyz_init2 - Weff2 * Mxyz_init1
        
        dot = Weff1 * Mxyz_init1 + Weff2 * Mxyz_init2 + Weff3 * Mxyz_init3
        
        Mxyz1 = (cos(phi) * Mxyz_init1 + sin(phi) * crs_prd1 + (1 - cos(phi)) * dot * Weff1)
        Mxyz2 = (cos(phi) * Mxyz_init2 + sin(phi) * crs_prd2 + (1 - cos(phi)) * dot * Weff2)
        Mxyz3 = (cos(phi) * Mxyz_init3 + sin(phi) * crs_prd3 + (1 - cos(phi)) * dot * Weff3)
    end
    @SVector [Mxyz1,Mxyz2,Mxyz3]
end

@inline function bloch2d_B1Ramp(M2d, M0, gx, gy, resol, pos, sim, w1, phiRF, off_res, tmp_acquire, B1_scale)
    @inbounds @views begin
        if ndims(B1_scale) .=== 2 
            for y = 1:pos[2]
                Threads.@threads for x = 1:pos[1]
                    count = 1 
                    for tt = 1:length(w1)
                        M0[x,y] = rotation_b1scale(M0[x,y], gx[tt], gy[tt], resol, sim[1][x], sim[2][y], w1[tt], phiRF[tt], off_res[x,y], B1_scale[x,y])
                        if tmp_acquire[tt] != 0
                            M2d[count,x,y] = M0[x,y]
                            count += 1
                        end
                    end
                end
            end
        elseif ndims(B1_scale) .=== 3
            for y = 1:pos[2]
                Threads.@threads for x = 1:pos[1]
                    count = 1 
                    for tt = 1:length(w1)
                        M0[x,y] = rotation_b1scale(M0[x,y], gx[tt], gy[tt], resol, sim[1][x], sim[2][y], w1[tt], phiRF[tt], off_res[x,y], B1_scale[tt,x,y])
                        if tmp_acquire[tt] != 0
                            M2d[count,x,y] = M0[x,y]
                            count += 1
                        end
                    end
                end
            end
        end
    end
end
# -----
# Multi Shot Sim
export Perform_Free_sim
@inline function Perform_Free_sim(B1_scale, gx_master, gy_master, fm, resol, nobj, Sim, pulse, tmp_acquire, num_avg;
    Object=ones(nobj[1], nobj[2])
)
    # Assumes a perfect 90
    @inbounds @views begin
        np = count(!iszero, tmp_acquire)

        # Need to change this to remove perfect 90
        M0 = [@SVector zeros(eltype(gx_master), 3) for i in 1:size(pulse, 1), j in 1:nobj[1], k in 1:nobj[2]]

        for j in 1:nobj[2], i in 1:nobj[1], k in 1:size(pulse, 1)
            M0[k,i,j] = @SVector [0.0,Object[i,j],0.0]
        end

        M2d_out = [@SVector zeros(eltype(gx_master), 3) for i in 1:size(pulse, 1), j in 1:np, k in 1:nobj[1], l in 1:nobj[2]]
        
        Threads.@threads for shot_index = 1:size(pulse, 1)
            bloch2d_B1Ramp(M2d_out[shot_index,:,:,:], M0[shot_index,:,:], gx_master, gy_master, resol, nobj, Sim, abs.(pulse[shot_index,:]), angle.(pulse[shot_index,:]), fm, tmp_acquire, B1_scale)
        end 
        
        Mxy_avg, Mxy_raw = Mxy_reshape_2d(M2d_out, Sim, num_avg)

       return Mxy_avg, Mxy_raw
    end
end

export Mxy_reshape_2d
function Mxy_reshape_2d(M2d_out, Sim, num_avg)
    @inbounds @views begin
        out = [M2d_out[i,j,k,l][1] + 1im .* M2d_out[i,j,k,l][2] for i in 1:size(M2d_out, 1), j in 1:size(M2d_out, 2), k in eachindex(Sim[1]), l in eachindex(Sim[2])]
        Mxy_out = zeros(Complex{eltype(M2d_out[1])}, size(M2d_out, 1), size(M2d_out, 2), Int(length(Sim[1]) / num_avg), Int(length(Sim[2]) / num_avg))

        for y in 1:Int(length(Sim[2]) / num_avg)
        Threads.@threads for x in 1:Int(length(Sim[1]) / num_avg)
            Mxy_out[:,:,x,y] = mean(mean(out[:,:,1 + num_avg * (x - 1):x * num_avg,1 + num_avg * (y - 1):y * num_avg], dims=3), dims=4)
            end
        end
    end
    return Mxy_out, out
end