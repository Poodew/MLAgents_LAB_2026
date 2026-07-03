using UnityEngine;
using UnityEngine.UIElements;

public class CartController : MonoBehaviour
{
    [SerializeField] Rigidbody cartRb;
    [SerializeField] Rigidbody poleRb;
    [SerializeField] HingeJoint pole;

    public float cartX;
    public float cartVel;
    public float poleAngle;
    public float poleAngVel;

    public float currentTime = 0;

    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {
        Application.targetFrameRate = 60;
        poleRb.rotation = Quaternion.Euler(0, 0, Random.Range(-5.0f, 5.0f));
    }

    // Update is called once per frame
    void FixedUpdate()
    {
        currentTime += Time.deltaTime;

        float k = Input.GetAxisRaw("Horizontal");

        cartX = cartRb.transform.position.x;
        cartVel = cartRb.linearVelocity.x;
        poleAngle = pole.angle;
        poleAngVel = poleRb.angularVelocity.z;

        if (IsEpisodeOver())
            ResetEpisode();

        cartRb.AddForce(Vector3.right * k * 2.5f);
    }

    public bool IsEpisodeOver()
    {
        if ((cartX < -1.0f || cartX > 1.0f) || (poleAngle < -80.0f || poleAngle > 80.0f))
        {
            Debug.Log(cartX);
            Debug.Log(poleAngle);
            return true;
        }

        return false;
    }

    public void ResetEpisode()
    {
        currentTime = 0;

        cartRb.position = Vector3.zero;
        poleRb.position = Vector3.zero;

        cartRb.rotation = Quaternion.identity;
        poleRb.rotation =
            Quaternion.Euler(0, 0, Random.Range(-5.0f, 5.0f));

        cartRb.linearVelocity = Vector3.zero;
        cartRb.angularVelocity = Vector3.zero;

        poleRb.linearVelocity = Vector3.zero;
        poleRb.angularVelocity = Vector3.zero;
    }
}
