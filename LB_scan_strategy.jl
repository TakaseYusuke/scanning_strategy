@inline function LB_track_TOD(
    nside::Integer,
    start::Integer,
    stop::Integer,
    smp_rate::Integer,
    FP_theta::Array,
    FP_phi::Array,
    start_point="pole",
)
    #= Compute the TOD of the hitpixel and the TOD of the detector orientation at a specified sampling rate from time start to stop. =#
    @inline function quaternion_rotator(omega, t, rotate_axis)
        #= Generate a quaternion that rotates by the angle omega*t around the rotate_axis axis. =#
        rot_ang = @views omega * t
        V = @views @SVector [cos(rot_ang/2), rotate_axis[1]*sin(rot_ang/2), rotate_axis[2]*sin(rot_ang/2), rotate_axis[3]*sin(rot_ang/2)]
        return Quaternion(V)
    end
    loop_times = @views ((stop - start) * smp_rate)  + smp_rate
    pix_tod = @views zeros(Int64, loop_times, length(FP_theta))
    psi_tod = @views zeros(Float64, loop_times, length(FP_theta))
    resol = @views Resolution(nside)
    alpha = @views deg2rad(45)
    beta = @views deg2rad(50)
    antisun_axis = @SVector [1, 0, 0]
    spin_axis = @SVector [1/√2, 0, 1/√2]
    y_axis = @SVector [0, 1, 0]
    z_axis = @SVector [0, 0, 1]
    if start_point =="pole"
        #= Set the initial position of the boresight to near the zenith. =#
        bore_0 = @views @SVector [sin((π/2)-alpha-beta), 0, cos((π/2)-alpha-beta)]
    end
    if start_point == "equator"
        #= Set the initial position of the boresight to near the equator. =#
        bore_0 = @views @SVector [cos((π/2)-alpha-beta), 0 , sin((π/2)-alpha-beta)]
    end
    #= Generate the quaternion at the initial position of the boresight and detector orientation. =#
    boresight_0 = @views Quaternion(0, bore_0)
    detector_orientation_0 = @views Quaternion(0, [-boresight_0.q3, 0, boresight_0.q1])
    q_det_gammma = @views quaternion_rotator(0, 1, bore_0)
    detector_orientation = @views q_det_gammma * detector_orientation_0 / q_det_gammma
    omega_revol = @views (2π) / (60 * 60 * 24 * 365)
    omega_prec = @views (2π / 60) / 192.348
    omega_spin = @views 0.05 * (2π / 60)
    for j in eachindex(FP_theta)
        #= 
        Generating the pointhing quaternion of other detectors in the focal plane 
        by adding an angular offset to the boresight based on the FP_theta and FP_phi information. 
        =#
        q_theta_in_FP = @views quaternion_rotator(deg2rad(FP_theta[j]), 1, y_axis)
        q_phi_in_FP = @views quaternion_rotator(deg2rad(FP_phi[j]), 1, bore_0)
        q_for_FP = @views q_phi_in_FP * q_theta_in_FP
        pointing = @views q_for_FP * boresight_0 / q_for_FP
        @inbounds @threads for i = 1:loop_times
            t = start + (i - 1) / smp_rate
            #= Generate the quaternion of revolution, precession, and spin at time t. =#
            q_revol = @views quaternion_rotator(omega_revol, t, z_axis)
            q_prec = @views quaternion_rotator(omega_prec, t, antisun_axis)
            q_spin = @views quaternion_rotator(omega_spin, t, spin_axis)
            #= 
            P has the value of the pointing vector (x,y,z) at time t as its imaginary part.
            The direction that the detector orientation is facing is then calculated as det_vec_t.
            =#
            Q = @views q_revol * q_prec * q_spin
            P =  @views Q * pointing / Q
            q_det = @views Q * detector_orientation / Q
            pointing_t = @views @SVector [P.q1, P.q2, P.q3]
            det_vec_t = @views @SVector [q_det.q1, q_det.q2, q_det.q3]
            #=
            The direction of movement can be calculated by the outer product of the pointhing vector and the detector orientaion vector.
            The vector of meridians can be calculated by combining the outer product of pointing and z axis.
            =#
            moving_direction = @views pointing_t × det_vec_t
            longitude = @views pointing_t × (pointing_t × z_axis)
            bore_vec = @views vec2ang(pointing_t[1], pointing_t[2], pointing_t[3])
            pix_tod[i, j] = @views ang2pixRing(resol, bore_vec[1], bore_vec[2])
            #=
            The vector standing perpendicular to the sphere is defined as the outer product of the meridian and the moving_direction. 
            The direction of divergent_vec depends on whether the trajectory enters the sphere from the left or the right side of the meridian.
            =#
            divergent_vec = @views longitude × moving_direction
            cosK = @views dot(moving_direction, longitude) / (norm(moving_direction) * norm(longitude))
            #=
            An if statement to prevent errors due to rounding errors that may result in |cosK|>1
            =#
            if abs(cosK) > 1.0
                cosK = sign(cosK)
            end
            psi_tod[i, j] = @views acos(cosK) * sign(divergent_vec[3]) * sign(pointing_t[3])
        end
    end
    return pix_tod, psi_tod
end
