using ReferenceFrameRotations
using Healpix
using LinearAlgebra
using Base.Threads
using StaticArrays

@inline function vec2ang_ver2(x, y, z)
    norm = sqrt(x^2 + y^2 + z^2)
    theta = acos(z / norm)
    phi = atan(y, x)
    if phi > π
        phi = π - phi
    end
    return (theta, phi)
end

@inline function LB_track_TOD(
    nside::Integer,
    start::Integer,
    stop::Integer,
    smp_rate::Integer,
    FP_theta::Array,
    FP_phi::Array,
    start_point,
)
    #= Compute the TOD of the hitpixel and the TOD of the detector orientation at a specified sampling rate from time start to stop. =#
    @inline function quaternion_rotator(omega, t, rotate_axis)
        #= Generate a quaternion that rotates by the angle omega*t around the rotate_axis axis. =#
        rot_ang = @views omega * t
        V = @views @SVector [cos(rot_ang/2.0), rotate_axis[1]*sin(rot_ang/2.0), rotate_axis[2]*sin(rot_ang/2.0), rotate_axis[3]*sin(rot_ang/2.0)]
        return Quaternion(V)
    end
    loop_times = @views ((stop - start) * smp_rate)  + smp_rate
    pix_tod = @views zeros(Int32, loop_times, length(FP_theta))
    psi_tod = @views zeros(Float32, loop_times, length(FP_theta))
    ##
    #ang_tod = @views zeros(Float32, 2, loop_times, length(FP_theta))
    ##

    #_m_ = Map{Float64,RingOrder}(nside)
    resol = @views Resolution(nside)
    alpha = @views deg2rad(45.0)
    beta = @views deg2rad(50.0)
    antisun_axis = @views @SVector [1.0, 0.0, 0.0]
    spin_axis = @views @SVector [1.0/√2.0, 0.0, 1.0/√2.0]
    y_axis = @views @SVector [0.0, 1.0, 0.0]
    z_axis = @views @SVector [0.0, 0.0, 1.0]
    omega_revol = @views (2π) / (60.0 * 60.0 * 24.0 * 365.25)#31557600.0
    omega_prec = @views (2π / 60.0) / 192.348
    omega_spin = @views 0.050 * (2π / 60)
    if start_point =="pole"
        #= Set the initial position of the boresight to near the zenith. =#
        bore_0 = @views @SVector [sin((π/2.0)-alpha-beta), 0, cos((π/2.0)-alpha-beta)]
    end
    if start_point == "equator"
        #= Set the initial position of the boresight to near the equator. =#
        bore_0 = @views @SVector [cos((π/2.0)-alpha-beta), 0 , sin((π/2.0)-alpha-beta)]
    end
    #= Generate the quaternion at the initial position of the boresight and detector orientation. =#
    boresight_0 = @views Quaternion(0.0, bore_0)
    detector_orientation_0 = @views Quaternion(0.0, [-boresight_0.q3, 0.0, boresight_0.q1])
    q_det_gammma = @views quaternion_rotator(0.0, 1.0, bore_0)
    detector_orientation = @views q_det_gammma * detector_orientation_0 / q_det_gammma

    @views @inbounds @simd for j in eachindex(FP_theta)
        #=
        Generating the pointhing quaternion of other detectors in the focal plane
        by adding an angular offset to the boresight based on the FP_theta and FP_phi information.
        =#
        q_theta_in_FP = quaternion_rotator(deg2rad(FP_theta[j]), 1, y_axis)
        q_phi_in_FP = quaternion_rotator(deg2rad(FP_phi[j]), 1, bore_0)
        q_for_FP = q_phi_in_FP * q_theta_in_FP
        pointing = q_for_FP * boresight_0 / q_for_FP
        @views @inbounds @threads for i = 1:loop_times
            t = start + (i - 1) / smp_rate
            #= Generate the quaternion of revolution, precession, and spin at time t. =#
            q_revol = quaternion_rotator(omega_revol, t, z_axis)
            q_prec = quaternion_rotator(omega_prec, t, antisun_axis)
            q_spin = quaternion_rotator(omega_spin, t, spin_axis)
            #=
            P has the value of the pointing vector (x,y,z) at time t as its imaginary part.
            The direction that the detector orientation is facing is then calculated as det_vec_t.
            =#
            Q = q_revol * q_prec * q_spin
            P = Q * pointing / Q
            q_det = Q * detector_orientation / Q
            pointing_t = @SVector [P.q1, P.q2, P.q3]
            det_vec_t = @SVector [q_det.q1, q_det.q2, q_det.q3]
            #=
            The direction of movement can be calculated by the outer product of the pointhing vector and the detector orientaion vector.
            The vector of meridians can be calculated by combining the outer product of pointing and z axis.
            =#
            moving_direction = pointing_t × det_vec_t
            longitude = pointing_t × (pointing_t × z_axis)
            bore_ang = vec2ang_ver2(pointing_t[1], pointing_t[2], pointing_t[3])

            #ang_tod[1, i, j] = bore_ang[1]
            #ang_tod[2, i, j] = bore_ang[2]

            pix_tod[i, j] = ang2pixRing(resol, bore_ang[1], bore_ang[2])

            #=
            The vector standing perpendicular to the sphere is defined as the outer product of the meridian and the moving_direction.
            The direction of divergent_vec depends on whether the trajectory enters the sphere from the left or the right side of the meridian.
            =#
            divergent_vec = longitude × moving_direction
            cosK = dot(moving_direction, longitude) / (norm(moving_direction) * norm(longitude))
            #=
            An if statement to prevent errors due to rounding errors that may result in |cosK|>1
            =#
            if abs(cosK) > 1.0
                cosK = sign(cosK)
            end
            psi_tod[i, j] = acos(cosK) * sign(divergent_vec[3]) * sign(pointing_t[3])
        end
    end
    return pix_tod, psi_tod
end


function Mapmaking(
    nside,
    times,
    smp_rate,
    FP_theta,
    FP_phi,
    start_point,
    )

    npix = nside2npix(nside)
    split_num = 6
    month = Int(times / split_num)
    #ω_HWP = 2.0 * π * (46.0 / 60.0)

    hit_map = zeros(Int32, npix)
    Cross = zeros(Float32, (2,4, npix))
    BEGIN = 0

    @inbounds @simd for i = 1:split_num
        println("process=", i, "/", split_num)
        END = i * month

        pix_tod, psi_tod = LB_track_TOD(nside, BEGIN, END, smp_rate, FP_theta, FP_phi, start_point)
        println("Start mapmaking!")

        @inbounds @simd for k = eachindex(pix_tod)
            bore = @views pix_tod[k]
            psi = @views psi_tod[k]
            TIME = @views BEGIN + (k - 1) / smp_rate

            hit_map[bore] += 1

            Cross[1,1,bore] += sin(psi)
            Cross[2,1,bore] += cos(psi)
            Cross[1,2,bore] += sin(2psi)
            Cross[2,2,bore] += cos(2psi)
            Cross[1,3,bore] += sin(3psi)
            Cross[2,3,bore] += cos(3psi)
            Cross[1,4,bore] += sin(4psi)
            Cross[2,4,bore] += cos(4psi)

        end
        BEGIN = END + 1
    end

    link1 = @views @. (Cross[1,1,:]/hit_map)^2 + (Cross[2,1,:]/hit_map)^2
    link2 = @views @. (Cross[1,2,:]/hit_map)^2 + (Cross[2,2,:]/hit_map)^2
    link3 = @views @. (Cross[1,3,:]/hit_map)^2 + (Cross[2,3,:]/hit_map)^2
    link4 = @views @. (Cross[1,4,:]/hit_map)^2 + (Cross[2,4,:]/hit_map)^2
    out_map = @views [hit_map, link1, link2, link3, link4]

    return out_map
end
