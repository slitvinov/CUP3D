from paraview.simple import *
import sys
import os

if len(sys.argv) < 3:
    sys.stderr.write("usage: pvpython pv.py LEVEL[,LEVEL...] FILE.xdmf2 [FILE.xdmf2 ...]\n")
    sys.exit(1)
levels = [float(s) for s in sys.argv[1].split(",")]
for path in sys.argv[2:]:
    reader = XDMFReader(FileNames=[path])
    reader.CellArrayStatus = ["chi", "vorticity"]
    merged = MergeBlocks(Input=reader)
    grid = CleantoGrid(Input=merged)
    calc = Calculator(Input=grid)
    calc.AttributeType = "Cell Data"
    calc.ResultArrayName = "omega"
    calc.Function = "mag(vorticity)"
    pts = CellDatatoPointData(Input=calc)
    pts.UpdatePipeline()
    b = pts.GetDataInformation().GetBounds()
    c = [(b[0] + b[1]) / 2, (b[2] + b[3]) / 2, (b[4] + b[5]) / 2]
    view = GetActiveViewOrCreate("RenderView")
    view.ViewSize = [1920, 1080]
    view.Background = [1, 1, 1]
    view.OrientationAxesVisibility = 0
    body = Contour(Input=pts)
    body.ContourBy = ["POINTS", "chi"]
    body.Isosurfaces = [0.5]
    bd = Show(body, view)
    bd.ColorArrayName = ["POINTS", ""]
    bd.DiffuseColor = [0.25, 0.25, 0.25]
    vort = Contour(Input=pts)
    vort.ContourBy = ["POINTS", "omega"]
    vort.Isosurfaces = levels
    vort.ComputeScalars = 1
    vd = Show(vort, view)
    ColorBy(vd, ("POINTS", "omega"))
    vd.Opacity = 0.35
    lut = GetColorTransferFunction("omega")
    lut.ApplyPreset("Cool to Warm", True)
    lut.RescaleTransferFunction(min(levels), max(levels))
    vd.SetScalarBarVisibility(view, True)
    Render(view)
    view.CameraFocalPoint = c
    view.CameraViewUp = [0, 0, 1]
    view.CameraPosition = [c[0] + 0.3 * (b[1] - b[0]), c[1] - 1.2 * (b[3] - b[2]), c[2] + 0.8 * (b[5] - b[4])]
    view.ResetCamera(False)
    Render(view)
    SaveScreenshot(os.path.splitext(path)[0] + ".pv.png", view)
    Delete(vd)
    Delete(bd)
    for p in (vort, body, pts, calc, grid, merged, reader):
        Delete(p)
