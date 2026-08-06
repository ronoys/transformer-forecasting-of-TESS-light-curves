"""Local entrypoint for building the TESS handoff file.

Use this for a quick interactive run. On Zaratan, prefer:
    sbatch tess.sh
"""

from zaratan_handoff import main


if __name__ == "__main__":
    main()
