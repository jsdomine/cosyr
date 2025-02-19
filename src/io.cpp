#include "io.h"

namespace cosyr {

/* -------------------------------------------------------------------------- */
IO::IO(Input& in_input,
       Beam& in_beam,
       Wavelets& in_wavelets,
       Mesh& in_mesh,
       Timer& in_timer)
  : input(in_input),
    beam(in_beam),
    wavelets(in_wavelets),
    mesh(in_mesh),
    timer(in_timer)
{}

/* -------------------------------------------------------------------------- */
void IO::dump(int i) {

  bool write_wavelets = input.wavelets.output and i >= input.wavelets.dump_start
                       and ((i - input.wavelets.dump_start) % input.wavelets.dump_interval == 0);

  bool write_beam = input.beam.output
                   and i >= input.beam.dump_start
                   and ((i - input.beam.dump_start) % input.beam.dump_interval == 0);

  bool write_mesh = input.mesh.output
                   and (i >= input.mesh.dump_start)
                   and ((i - input.mesh.dump_start) % input.mesh.dump_interval == 0);

  bool interpolate = ((i >= input.remap.start_step)
                     and ((i - input.remap.start_step) % input.remap.interval == 0
                       or i == input.kernel.num_step - 1));

//  #define DEBUG
  #ifdef DEBUG
    std::cout << input.mpi.rank
              << ": step = " << i <<", write_wavelets ? "
              << std::boolalpha << write_wavelets << std::endl;
  #endif

  if (write_wavelets) {
    timer.start("write_wavelet");
    dump_wavelets(i);
    timer.stop("write_wavelet");
  }

  #ifdef DEBUG
    std::cout << input.mpi.rank << ": finished dumping wavelets" << std::endl;
  #endif

  #ifdef DEBUG
    std::cout << input.mpi.rank
              << ": step = " << i <<", write_beam ? "
              << std::boolalpha << write_beam << std::endl;
  #endif

  if (write_beam) {
    timer.start("write_beam");
    dump_particles(i);
    timer.stop("write_beam");
  }

  if (interpolate) {
    timer.start("mesh_sync");
    mesh.sync();
    timer.stop("mesh_sync");

    if (write_mesh) {
      timer.start("write_mesh");
      dump_mesh(i);
      timer.stop("write_mesh");
    }
  }
}

/* -------------------------------------------------------------------------- */
void IO::dump_wavelets(int step) const {

    #ifdef DEBUG
      std::cout << input.mpi.rank
                << ": " << " enter dump_wavelets." << std::endl;
    #endif

  if (input.mpi.rank == 0) { std::cout << "dump wavelets ... " << std::flush; }

  int const extent = std::min(input.wavelets.num_wavelet_files,
                              input.kernel.num_particles);

  for (int index_particle = 0; index_particle < extent; index_particle++) {

    std::ofstream file;
    std::string suffix(std::to_string(index_particle) + "_" + std::to_string(input.mpi.rank) + ".csv");
    std::string wavefront_file(input.kernel.run_name + "/wavelet/" + std::to_string(step) + "/wavefronts_" + suffix);
    std::string trajectory_file(input.kernel.run_name + "/traj/" + std::to_string(step) + "/trajectory_" + suffix);
    std::string field_file(input.kernel.run_name + "/wavelet/" + std::to_string(step) + "/field_" + suffix);

    int const num_emitted = input.kernel.num_wavefronts * input.kernel.num_dirs;
    int const num_loaded = input.wavelets.count;
    auto num_active = wavelets.transfer_to_host();

    // Writes wavefronts position of specified particle into file
    file.open(wavefront_file);

    if (file.good()) {
      file << "# pos_wave_x, pos_wave_y" << "\n";

      // step 2: dump loaded wavelets
      if (num_loaded > 0) {
        int const start_loaded = index_particle * num_loaded;
        int const extent_loaded = start_loaded + num_loaded;

        for (int j = start_loaded; j < extent_loaded; j++) {
          file << std::setprecision(12);
          file << wavelets.loaded.host.coords(j, WT_POS_X) << ","
               << wavelets.loaded.host.coords(j, WT_POS_Y) << "\n";
        }
      }

      // step 1: dump emitted wavelets
      if (num_emitted > 0) {
        int const start_emit = index_particle * num_emitted;
        int const extent_emit = start_emit + num_active[index_particle] - num_loaded;

        for (int j = start_emit; j < extent_emit; j++) {
          file << std::setprecision(12);
          file << wavelets.emitted.host.coords(j, WT_POS_X) << ","
               << wavelets.emitted.host.coords(j, WT_POS_Y) << "\n";
        }
      }
    } else
      throw std::runtime_error("failed to open '" + wavefront_file + "'");

    file.close();

    #ifdef DEBUG
      std::cout << input.mpi.rank
                << ": " << num_loaded + num_emitted << " wavelets written." << std::endl;
    #endif

    // Write the particle trajectory into file
    file.open(trajectory_file);

    if (file.good()) {
      file << "# traj_x, traj_y" << "\n";

      int const start_traject = index_particle * input.kernel.num_step;
      int const extent_traject = start_traject + step + 1;

      for (int j = start_traject; j < extent_traject; j++) {
        int const k = j * NUM_TRAJ_QUANTITIES;
        file << std::setprecision(12);
        file << beam.trajectory[k + TRAJ_POS_X] << ","
             << beam.trajectory[k + TRAJ_POS_Y] << "\n";
      }
    } else
      throw std::runtime_error("failed to open '" + trajectory_file + "'");

    file.close();

    // Write the velocity ,acceleration and total field into file
    file.open(field_file);
    if (file.good()) {

      file << "# field_1, field_2, field_3" << "\n";

      // step 2: dump loaded wavelets
      if (num_loaded > 0) {
        int const start_loaded = index_particle * num_loaded;
        int const extent_loaded = start_loaded + num_loaded;

        for (int j = start_loaded; j < extent_loaded; j++) {
          file << std::setprecision(12);
          for (int k = 0; k <= DIM; ++k) {
            file << wavelets.loaded.host.fields[k](j) << (k < DIM ? ", " : "\n");
          }
        }
      }

      // step 1: dump emitted wavelets
      if (num_emitted > 0) {
        int const start_emit = index_particle * num_emitted;
        int const extent_emit = start_emit + num_active[index_particle] - num_loaded;

        for (int j = start_emit; j < extent_emit; j++) {
          file << std::setprecision(12);
          for (int k = 0; k <= DIM; ++k) {
            file << wavelets.emitted.host.fields[k](j) << (k < DIM ? ", " : "\n");
          }
        }
      }
    } else
      throw std::runtime_error("failed to open '" + field_file + "'");

    file.close();
  }

  if (input.mpi.rank == 0) { std::cout << "done" << std::endl; }
}

/* -------------------------------------------------------------------------- */
void IO::dump_particles(int step) const {

  if (input.mpi.rank == 0) { std::cout << "dump beam ... " << std::flush; }

  std::ofstream file;
  std::string suffix(std::to_string(step) + "_" + std::to_string(input.mpi.rank) + ".csv");
  std::string particle_file(input.kernel.run_name + "/beam/" + std::to_string(step) + "/particles_" + suffix);

  file.open(particle_file);

  if (file.good()) {

    file << "# pos_x, pos_y, mom_x, mom_y, gamma" << "\n";

    auto position = Cabana::slice<Beam::Position>(beam.particles);
    auto momentum = Cabana::slice<Beam::Momentum>(beam.particles);
    auto gamma    = Cabana::slice<Beam::Gamma>(beam.particles);

    for (int j = 0; j < input.kernel.num_particles; j++) {
      file << std::setprecision(12);
      file << position(j, PART_POS_X) << "," << position(j, PART_POS_Y) << ","
           << momentum(j, PART_MOM_X) << "," << momentum(j, PART_MOM_Y) << ","
           << gamma(j) << "\n";
    }
  } else
    throw std::runtime_error("failed to open '" + particle_file + "'");

  file.close();

  if (input.mpi.rank == 0) { std::cout << "done" << std::endl; }
}

/* -------------------------------------------------------------------------- */
void IO::dump_mesh(int step) const {

  // only for the master rank
  if (input.mpi.rank == 0) { std::cout << "dump mesh ... " << std::flush; }
  else { return; }

  std::ofstream file;
  std::string position_file(input.kernel.run_name + "/mesh/" + std::to_string(step) + "/comoving_mesh_pos.csv");
  std::string field_file(input.kernel.run_name + "/mesh/" + std::to_string(step) + "/comoving_mesh_field.csv");

  file.open(position_file);

  if (file.good()) {
    auto x = Cabana::slice<X>(mesh.points);
    auto y = Cabana::slice<Y>(mesh.points);

    file << "# pos_x, pos_y" << "\n";

    for (int i = 0; i < mesh.num_points; i++) {
      file << std::setprecision(12);
      file << x(i) << "," << y(i) << "\n";
    }
  } else
    throw std::runtime_error("failed to open '" + position_file + "'");

  file.close();
  file.open(field_file);

  if (file.good()) {

    auto field_1 = Cabana::slice<F1>(mesh.fields);
    auto field_2 = Cabana::slice<F2>(mesh.fields);
    auto field_3 = Cabana::slice<F3>(mesh.fields);

    file << "# field_1, field_2, field_3" << "\n";

    for (int i = 0; i < mesh.num_points; i++) {
      file << std::setprecision(12);
      file << field_1(i) << ", " << field_2(i) << ", " << field_3(i) << "\n";
    }
  } else
    throw std::runtime_error("failed to open '" + field_file + "'");

  file.close();

  if (input.mpi.rank == 0) { std::cout << "done" << std::endl; }
}

/* -------------------------------------------------------------------------- */
} // namespace cosyr
