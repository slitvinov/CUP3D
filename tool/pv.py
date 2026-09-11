from paraview.simple import *
import sys
import os

for path in sys.argv[1:]:
    reader = XDMFReader(FileNames=[path])
    reader.CellArrayStatus = ["chi", "vorticity"]
    merged = MergeBlocks(Input=reader)
    grid = CleantoGrid(Input=merged)
    calc = Calculator(Input=grid)
    calc.AttributeType = "Cell Data"
    calc.ResultArrayName = "omega"
    calc.Function = "mag(vorticity)"
    calc.UpdatePipeline()
    b = calc.GetDataInformation().GetBounds()
    omax = calc.CellData["omega"].GetRange()[1]
    view = GetActiveViewOrCreate("RenderView")
    view.ViewSize = [1600, 800]
    view.Background = [1, 1, 1]
    view.OrientationAxesVisibility = 0
    plane = Slice(Input=calc)
    plane.SliceType = "Plane"
    plane.SliceType.Origin = [(b[0] + b[1]) / 2, (b[2] + b[3]) / 2, (b[4] + b[5]) / 2]
    plane.SliceType.Normal = [0, 0, 1]
    sd = Show(plane, view)
    ColorBy(sd, ("CELLS", "omega"))
    lut = GetColorTransferFunction("omega")
    lut.ApplyPreset("Cool to Warm", True)
    lut.RescaleTransferFunction(0.0, omax)
    sd.SetScalarBarVisibility(view, True)
    body = CellDatatoPointData(Input=grid)
    iso = Contour(Input=body)
    iso.ContourBy = ["POINTS", "chi"]
    iso.Isosurfaces = [0.5]
    fd = Show(iso, view)
    fd.ColorArrayName = ["POINTS", ""]
    fd.DiffuseColor = [0.2, 0.2, 0.2]
    view.InteractionMode = "2D"
    view.CameraPosition = [(b[0] + b[1]) / 2, (b[2] + b[3]) / 2, b[5] + 10]
    view.CameraFocalPoint = [(b[0] + b[1]) / 2, (b[2] + b[3]) / 2, (b[4] + b[5]) / 2]
    view.CameraViewUp = [0, 1, 0]
    view.ResetCamera()
    Render(view)
    SaveScreenshot(os.path.splitext(path)[0] + ".pv.png", view)
    Delete(fd)
    Delete(sd)
    for p in (iso, body, plane, calc, grid, merged, reader):
        Delete(p)
