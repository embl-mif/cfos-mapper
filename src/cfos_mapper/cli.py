"""
Command line: `cfos-mapper <step> ...` (or `python -m cfos_mapper <step> ...`).

    cfos-mapper init FOLDER                         write template settings.json + regions.json
    cfos-mapper napari                              open napari with the four widgets
    cfos-mapper prepare O15 regionA_cortex          cut a test region + find its candidates
    cfos-mapper prepare-all                         ... every test region that has no candidates yet
    cfos-mapper predict O15 regionA_cortex network_v2   a network's calls on a test region
    cfos-mapper predict-all network_v2              ... on every test region
    cfos-mapper train network_v3 --epochs 150 --gpu 1   train a network on all annotations
    cfos-mapper run-brain O15 --network network_v3 --gpu 0   cells in a whole brain
    cfos-mapper register O15                        register a brain to the atlas (brainreg)
    cfos-mapper count O15 --network network_v3      cells per atlas region
    cfos-mapper summary --network network_v3        cells per region, all brains in one table

Every command takes --settings FILE (default: $CFOS_MAPPER_SETTINGS or ./settings.json).
"""

import argparse
import sys


def parser():
    p = argparse.ArgumentParser(prog="cfos-mapper", description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("--settings", help="settings file (default: $CFOS_MAPPER_SETTINGS or ./settings.json)")
    p.add_argument("--version", action="store_true", help="print the version and stop")
    # --settings is also accepted after the step name
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--settings", default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    subparsers = p.add_subparsers(dest="step")

    class Steps:
        @staticmethod
        def add_parser(name, **kwargs):
            return subparsers.add_parser(name, parents=[common], **kwargs)

    steps = Steps()

    s = steps.add_parser("init", help="write template settings.json and regions.json into a folder")
    s.add_argument("folder")
    steps.add_parser("napari", help="open napari with the cfos-mapper widgets")

    s = steps.add_parser("prepare", help="cut a test region out of a brain and find its candidates")
    s.add_argument("brain")
    s.add_argument("region")
    steps.add_parser("prepare-all", help="prepare every test region that is not prepared yet")

    s = steps.add_parser("predict", help="a network's calls on a test region (+ random-spot control)")
    s.add_argument("brain")
    s.add_argument("region")
    s.add_argument("network", help="'pretrained' or a network in the networks folder")
    s = steps.add_parser("predict-all", help="a network's calls on every test region")
    s.add_argument("network")

    s = steps.add_parser("train", help="train a new network from all saved annotations")
    s.add_argument("name", help="name of the new network, e.g. network_v3")
    s.add_argument("--epochs", type=int, default=60)
    s.add_argument("--test-fraction", type=float, default=0.2)
    s.add_argument("--gpu", help="GPU to use (0, 1, ...)")

    s = steps.add_parser("run-brain", help="find the cells in a whole brain")
    s.add_argument("brain")
    s.add_argument("--network", required=True)
    s.add_argument("--gpu", help="GPU to use (0, 1, ...)")
    s.add_argument("--planes", nargs=2, type=int, metavar=("FIRST", "LAST"),
                   help="only these z-planes (quick test)")

    s = steps.add_parser("register", help="register a brain to the atlas with brainreg")
    s.add_argument("brain")
    s.add_argument("--orientation", help="override settings atlas.orientation, e.g. sal")
    s.add_argument("--atlas", help="override settings atlas.name, e.g. allen_mouse_25um")

    s = steps.add_parser("count", help="count the cells of a whole-brain run per atlas region")
    s.add_argument("brain")
    s.add_argument("--network", required=True)
    s.add_argument("--run", help="run folder name if not the network name (e.g. network_v3_planes900-1000)")

    s = steps.add_parser("summary", help="cells per region of all brains in one table")
    s.add_argument("--network", required=True)
    s.add_argument("--brains", nargs="*", help="only these brains")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.version:
        from . import __version__
        print(__version__)
        return 0
    if args.step is None:
        parser().print_help()
        return 1

    if args.step == "init":
        from .settings import write_templates
        for file in write_templates(args.folder):
            print(f"written {file}")
        print("Edit settings.json (brains_folder, results_folder), then run commands with "
              f"--settings {args.folder}/settings.json")
        return 0

    # GPU steps: choose the GPU and load cuDNN before torch does
    from .gpu import load_gpu_libraries, use_gpu
    use_gpu(getattr(args, "gpu", None))
    if args.step in ("prepare", "prepare-all", "predict", "predict-all", "train", "run-brain"):
        load_gpu_libraries()

    import cfos_mapper as cf

    settings = cf.load_settings(args.settings)
    if args.step == "napari":
        import os

        import napari

        from .plugin import open_viewer
        os.environ["CFOS_MAPPER_SETTINGS"] = settings["settings_file"]  # for the widgets
        open_viewer()
        napari.run()
    elif args.step == "prepare":
        cf.prepare_region(args.brain, args.region, settings)
    elif args.step == "prepare-all":
        for brain, boxes in cf.load_regions(settings).items():
            for region in boxes:
                if not cf.region_ready(brain, region, settings):
                    print(f"preparing {brain} {region}", flush=True)
                    cf.prepare_region(brain, region, settings)
    elif args.step == "predict":
        cf.get_predictions(args.brain, args.region, args.network, settings)
    elif args.step == "predict-all":
        for brain, boxes in cf.load_regions(settings).items():
            for region in boxes:
                print(f"{args.network}: {brain} {region}", flush=True)
                cf.get_predictions(brain, region, args.network, settings)
    elif args.step == "train":
        cf.train_network(args.name, settings, epochs=args.epochs, test_fraction=args.test_fraction)
    elif args.step == "run-brain":
        cf.run_brain(args.brain, args.network, settings, planes=args.planes)
    elif args.step == "register":
        if args.orientation:
            settings["atlas"]["orientation"] = args.orientation
        if args.atlas:
            settings["atlas"]["name"] = args.atlas
        cf.register_brain(args.brain, settings)
    elif args.step == "count":
        cf.count_cells_per_region(args.brain, args.network, settings, run=args.run)
    elif args.step == "summary":
        table = cf.summary_table(args.network, settings, brains=args.brains)
        print(f"{len(table)} regions -> {settings['reports_folder']}/{args.network}/cells_per_region_all_brains.csv")
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
