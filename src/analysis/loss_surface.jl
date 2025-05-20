LerpNSVec(α, NSVec1, NSVec2) = (1 - α) .* NSVec1 + α .* NSVec2
PlaneOriginNSVec(s, t, e1, e2) = s * e1 + t * e2
PlaneNSVec(s, t, e1, e2, p0) = p0 + PlaneOriginNSVec(s, t, e1, e2)
function NormalOfPlane(e1, e2)
    B = hcat(e1, e2)
    n = nullspace(B')[:, end] # taking only first
    n = normalize(n)
    return n
end
