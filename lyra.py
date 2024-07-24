#!/usr/bin/python3
# -*- coding: utf-8 -*-

import argparse
from lyra.vibration import calculate_frequency, electronic_vibrational_AH, electronic_vibrational_VG
from lyra.utils import write_freq_modes


def _build_parser():
    parser = argparse.ArgumentParser(
        prog="Lyra",
        description="Calculating Electron-Phonon Coupling"
    )
    subcmd = parser.add_subparsers(
        dest="subcmd", help="subcommands", metavar="SUBCOMMAND"
    )
    subcmd.required = True

    vibro_parser = subcmd.add_parser(
        "evc", help="Electronic Vibrational Coupling Analysis")
    vibro_parser.add_argument("initial",
                              help="The fchk file containing Hessian of initial state")
    vibro_parser.add_argument("--AH",
                              help="Adiabatic Hessian, requires the fchk file containing Hessian of final state")
    vibro_parser.add_argument("--VG",
                              help="Vertical Gradient, requires the fchk file containing Gradient of final state")

    freq_parser = subcmd.add_parser("freq", help="Frequency Analysis")
    freq_parser.add_argument("fchk",
                             help="The fchk file containing Hessian")

    return parser


if __name__ == "__main__":
    parser = _build_parser()
    args = parser.parse_args()

    if args.subcmd == 'evc':
        if args.AH is not None and args.VG is not None:
            raise RuntimeError("Contradictory input of final state.")
        if args.AH is None and args.VG is None:
            raise RuntimeError("The input of final state is missed.")
        if args.AH:
            electronic_vibrational_AH(args.initial, args.AH)
        else:
            electronic_vibrational_VG(args.initial, args.VG)
    elif args.subcmd == 'freq':
        freq, modes = calculate_frequency(args.fchk)
        write_freq_modes(freq, modes)
